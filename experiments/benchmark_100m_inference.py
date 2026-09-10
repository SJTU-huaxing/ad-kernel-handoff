"""Paired real-prompt AD64 / official FLA Hedgehog inference benchmark.

No optimizer, gradients, training, or checkpoint changes. Run from repository
root in nonlinear-qk with PYTHONPATH=src. Default device is logical NPU 0.
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import numpy as np
import torch
import torch_npu
import fla
from torch.nn import functional as F
from ad_kernel.attention import numerical_split_count
from pretrain_100m.common import ROOT, DEFAULT_CONFIG, atomic_json, digest, source_binding
from pretrain_100m.model import LanguageModel, ModelConfig
from inference_100m_runtime import Runtime, GraphDecoder


def tensor_hash(tensors):
    h = hashlib.sha256()
    for name, value in tensors:
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def error(actual, expected):
    a, e = actual.detach().cpu().double(), expected.detach().cpu().double()
    return {"max_abs": (a-e).abs().max().item(),
            "relative_l2": ((a-e).norm()/e.norm()).item()}


def assert_error(result, *, fp32=False):
    # BF16 backbone has rounding differences between GEMV and batched GEMM.
    if fp32:
        assert result["relative_l2"] < 1e-4 and result["max_abs"] < 1e-4, result
    else:
        assert result["relative_l2"] < .01 and result["max_abs"] <= .0625, result


def validate(runtime, inputs):
    tokens = inputs[:, :73]
    # Isolate cache algebra from the different BF16 GEMV/GEMM rounding paths.
    fp32_rows = []
    with torch.autocast("npu", enabled=False):
        fp32_reference = F.linear(runtime.model.hidden_states(tokens), runtime.model.embedding.weight)
        _, fp32_cache = runtime.prefill(tokens[:, :65])
        for i in range(65, 73):
            fp32_logits, fp32_cache = runtime.step(tokens[:, i], fp32_cache)
            result = error(fp32_logits, fp32_reference[:, i])
            assert_error(result, fp32=True)
            fp32_rows.append({"position": i, **result})
        del fp32_reference, fp32_cache, fp32_logits
    reference = F.linear(runtime.model.hidden_states(tokens), runtime.model.embedding.weight)
    direct, _ = runtime.prefill(tokens, all_logits=True)
    prefill_error = error(direct, reference)
    assert_error(prefill_error)
    _, base = runtime.prefill(tokens[:, :65])
    graph = GraphDecoder(runtime, tokens[:, 65], base)
    eager = base.clone()
    rows = []
    for i in range(65, 73):
        e, eager = runtime.step(tokens[:, i], eager)
        g = graph.step(tokens[:, i])
        graph_error, prefix_error = error(g, e), error(e, reference[:, i])
        assert_error(graph_error)
        assert_error(prefix_error)
        rows.append({"position": i, "graph_vs_eager": graph_error, "cache_vs_full_sequence": prefix_error})
    assert graph.position.item() == 73
    state_errors = []
    for gs, es in zip(graph.cache.layers, eager.layers):
        for key in ["kv", "z", "log_scale"]:
            result = error(getattr(gs, key), getattr(es, key))
            assert result["relative_l2"] < 1e-5, result
            state_errors.append(result)
    assert base.tensor_bytes() == eager.tensor_bytes() == graph.cache.tensor_bytes()
    return {"fp32_cache_vs_full_sequence": fp32_rows,
            "prefill_vs_original_model": prefill_error, "steps": rows,
            "graph_state_errors": state_errors, "actual_final_position": 73,
            "cache_bytes": base.tensor_bytes(), "passed": True}


def timed(fn):
    torch.npu.synchronize()
    start_event, end_event = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    start = time.perf_counter()
    start_event.record()
    result = fn()
    end_event.record()
    end_event.synchronize()
    return result, {"wall_ms": (time.perf_counter()-start)*1000,
                    "event_ms": start_event.elapsed_time(end_event)}


def summary(samples, token_count, steps=1):
    wall = [s["wall_ms"] for s in samples]
    device = [s["event_ms"] for s in samples]
    return {"median_wall_ms": statistics.median(wall), "median_event_ms": statistics.median(device),
            "median_ms_per_step": statistics.median(wall)/steps,
            "tokens_per_second": token_count/(statistics.median(wall)/1000), "samples": samples}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", nargs="+", default=["ad64", "hedgehog"], choices=["ad64", "hedgehog"])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 8])
    ap.add_argument("--prefill-lengths", nargs="+", type=int, default=[512, 2048, 8192])
    ap.add_argument("--decode-contexts", nargs="+", type=int, default=[512, 2048, 4096])
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT/"work/pretrain_100m_2b/inference_ad_vs_hedgehog")
    args = ap.parse_args()
    assert max(args.decode_contexts) + args.decode_tokens <= 8192
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / ("correctness.json" if args.check_only else "benchmark.json")
    if target.exists():
        raise FileExistsError(f"Use a new --out to preserve existing measurements: {target}")
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    config = json.loads(DEFAULT_CONFIG.read_text())
    test_file = ROOT/config["data"]["directory"]/"test.bin"
    stream = np.memmap(test_file, dtype="<u2", mode="r")
    stride = 8193
    array = np.stack([np.asarray(stream[i*stride:(i+1)*stride], dtype=np.int64)
                      for i in range(max(args.batches))])
    inputs = torch.from_numpy(array).to(f"npu:{args.device}")
    sources = source_binding()
    for name in ["benchmark_100m_inference.py", "inference_100m_runtime.py"]:
        sources[f"experiments/{name}"] = digest(ROOT/"experiments"/name)
    result = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "arguments": {**vars(args), "out": str(args.out)}, "source_sha256": sources,
              "protocol_sha256": digest(DEFAULT_CONFIG), "test_file_sha256": digest(test_file),
              "input_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
              "torch": str(torch.__version__), "torch_npu": str(torch_npu.__version__),
              "ascend_visible_devices": os.environ.get("ASCEND_VISIBLE_DEVICES"),
              "npu_snapshot_before": subprocess.check_output(["npu-smi", "info"], text=True),
              "scope": "Fresh same-seed 100M models, no trained-checkpoint quality claim. Single NPU, BF16 autocast backbone; FP32 weights/features/state. FLA fused RMSNorm and SwiGLU. Includes all 12 blocks, final norm, last-token 50257-vocabulary head and compact cache. Prompt GPU/NPU-resident. Decode consumes identical held-out tokens, excludes sampling/network/tokenization/prefill. Timed graph token device copy and actual state/position advance included. Warmup, graph compilation/capture and state reset excluded. Synchronized wall time primary; event intervals also recorded. No training or backward.",
              "models": {}, "correctness": [], "rows": []}
    backbone_hash = None
    for round_index in range(1 if args.check_only else args.rounds):
        order = args.methods if round_index % 2 == 0 else list(reversed(args.methods))
        for method in order:
            torch.manual_seed(args.seed)
            model = LanguageModel(ModelConfig(**config["architecture"], method=method, feature_seed=args.seed))
            common_hash = tensor_hash((n, p) for n, p in model.named_parameters() if ".features." not in n)
            if backbone_hash is None: backbone_hash = common_hash
            assert backbone_hash == common_hash
            result["models"][method] = {"parameter_budget": model.parameter_budget(), "backbone_sha256": common_hash,
                                         "weights_sha256": tensor_hash(model.state_dict().items())}
            model.to(f"npu:{args.device}").eval()
            runtime = Runtime(model)
            with torch.inference_mode(), torch.autocast("npu", dtype=torch.bfloat16):
                for batch in args.batches:
                    tokens = inputs[:batch]
                    if round_index == 0:
                        check = validate(runtime, tokens)
                        result["correctness"].append({"method": method, "batch": batch, **check})
                        print(json.dumps({"validated": method, "batch": batch, "passed": True}), flush=True)
                        atomic_json(target, result)
                    if args.check_only: continue
                    for length in args.prefill_lengths:
                        prompt = tokens[:, :length]
                        for _ in range(3): runtime.prefill(prompt)
                        times, splits, peaks = [], [], []
                        for _ in range(args.samples):
                            torch.npu.reset_peak_memory_stats()
                            before = numerical_split_count()
                            (logits, cache), sample = timed(lambda: runtime.prefill(prompt))
                            assert torch.isfinite(logits).all().item()
                            times.append(sample)
                            splits.append(numerical_split_count()-before)
                            peaks.append(torch.npu.max_memory_allocated())
                            cache_bytes = cache.tensor_bytes()
                            del logits, cache
                        row = {"round": round_index, "method": method, "batch": batch, "length": length,
                               "stage": "prefill", "execution": "eager", **summary(times, batch*length),
                               "cache_bytes": cache_bytes, "peak_allocated_bytes": max(peaks), "numerical_splits": splits}
                        result["rows"].append(row)
                        atomic_json(target, result)
                        print(json.dumps({k:v for k,v in row.items() if k != "samples"}), flush=True)
                    _, base = runtime.prefill(tokens[:, :args.decode_contexts[0]])
                    graph = GraphDecoder(runtime, tokens[:, args.decode_contexts[0]], base)
                    del base
                    for length in args.decode_contexts:
                        _, base = runtime.prefill(tokens[:, :length])
                        # Materialize token views outside timing for both modes.
                        future = list(tokens[:, length:length+args.decode_tokens].unbind(1))
                        for execution in (["eager", "graph"] if round_index % 2 == 0 else ["graph", "eager"]):
                            times, peaks = [], []
                            for iteration in range(args.samples+1):
                                if execution == "graph": graph.reset(base)
                                else: eager_cache = base.clone()
                                def decode():
                                    nonlocal eager_cache
                                    for token in future:
                                        if execution == "graph": logits = graph.step(token)
                                        else: logits, eager_cache = runtime.step(token, eager_cache)
                                    return logits
                                torch.npu.reset_peak_memory_stats()
                                logits, sample = timed(decode)
                                assert torch.isfinite(logits).all().item()
                                final_position = graph.position.item() if execution == "graph" else eager_cache.position
                                final_bytes = graph.cache.tensor_bytes() if execution == "graph" else eager_cache.tensor_bytes()
                                assert final_position == length+args.decode_tokens and final_bytes == base.tensor_bytes()
                                if iteration:
                                    times.append(sample)
                                    peaks.append(torch.npu.max_memory_allocated())
                                del logits
                            row = {"round": round_index, "method": method, "batch": batch, "length": length,
                                   "stage": "decode", "execution": execution,
                                   **summary(times, batch*args.decode_tokens, args.decode_tokens),
                                   "cache_bytes": base.tensor_bytes(), "cache_bytes_after_decode": final_bytes,
                                   "final_position": final_position, "peak_allocated_bytes": max(peaks)}
                            result["rows"].append(row)
                            atomic_json(target, result)
                            print(json.dumps({k:v for k,v in row.items() if k != "samples"}), flush=True)
                        del base, eager_cache
                    del graph
            del runtime, model
            gc.collect()
            torch.npu.empty_cache()
    result["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result["complete"] = True
    atomic_json(target, result)
    print(f"Saved {target}", flush=True)


if __name__ == "__main__": main()
