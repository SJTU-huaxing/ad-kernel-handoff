"""Exclusive-device cache/prefill/decode measurements at the final fixed budget."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import psutil
import torch
import torch_npu
import fla
from ad_kernel.attention import numerical_split_count
from ad_kernel.model import LinearLanguageModel, ModelConfig
from inference_runtime import LMRuntime

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "work/pretraining"
PROTOCOL = ROOT / "configs/inference_benchmark_v1.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def guard_exclusive():
    blocked = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        argv = process.info["cmdline"] or []
        if any(Path(arg).name in {"pretrain.py", "evaluate_pretrained.py", "benchmark_backends.py", "probe_full_model.py"} for arg in argv):
            blocked.append({"pid": process.info["pid"], "command": argv})
    if blocked:
        raise RuntimeError(f"Training/quality/other benchmark jobs are active; timing would be concurrent: {blocked}")


def summarize(samples, tokens):
    median = statistics.median(samples)
    return {"median_seconds": median, "samples_seconds": samples, "tokens_per_second": tokens / median}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--device", required=True)
    ap.add_argument("--batch", type=int, choices=[1, 4], required=True)
    args = ap.parse_args()
    guard_exclusive()
    protocol = json.loads(PROTOCOL.read_text())
    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    training = checkpoint["binding"]
    assert not training["synthetic"] and training["seed"] == 11 and checkpoint["step"] == 16384
    assert training["protocol_sha256"] == digest(ROOT / "configs/pretraining_protocol_v2.json")
    for source, expected in training["code_sha256"].items():
        assert digest(ROOT / source) == expected
    metadata = json.loads((BASE / "data/fineweb_edu_v1/manifest.json").read_text())
    test_file = BASE / "data/fineweb_edu_v1/test.bin"
    assert digest(test_file) == metadata["artifacts"]["test.bin"]["sha256"]
    sources = [Path(__file__), ROOT / "experiments/inference_runtime.py", PROTOCOL]
    code_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    code_id = hashlib.sha256(json.dumps(code_hashes, sort_keys=True).encode()).hexdigest()[:12]
    out = BASE / "efficiency" / f"{training['method']}_s11_step016384" / code_id
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"batch{args.batch}.json"
    binding = {"checkpoint_sha256": digest(args.checkpoint), "training": training, "benchmark_code_sha256": code_hashes,
               "device": args.device, "batch": args.batch, "test_file_sha256": metadata["artifacts"]["test.bin"]["sha256"]}
    if target.exists():
        assert json.loads(target.read_text())["binding"] == binding
        print("Verified existing inference benchmark", flush=True)
        return
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    model = LinearLanguageModel(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.to(args.device).eval()
    runtime = LMRuntime(model)
    stream = np.memmap(test_file, dtype="<u2", mode="r")
    stride = max(protocol["lengths"]) + protocol["decode_tokens"] + 1
    inputs = torch.from_numpy(np.stack([np.asarray(stream[i * stride:(i + 1) * stride], dtype=np.int64) for i in range(args.batch)])).to(args.device)
    snapshot = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True).stdout
    results = []
    repetitions = protocol["warmups"] + protocol["measurements"]
    with torch.inference_mode(), torch.autocast("npu", dtype=torch.bfloat16):
        for length in protocol["lengths"]:
            guard_exclusive()
            prefill, decode, prefill_peaks, decode_peaks, split_counts = [], [], [], [], []
            for iteration in range(repetitions):
                torch.npu.synchronize()
                torch.npu.reset_peak_memory_stats()
                before = numerical_split_count()
                start = time.perf_counter()
                logits, cache = runtime.prefill(inputs[:, :length])
                torch.npu.synchronize()
                elapsed = time.perf_counter() - start
                cache_bytes = cache.tensor_bytes()
                peak = torch.npu.max_memory_allocated()
                assert torch.isfinite(logits).all().item()
                if iteration >= protocol["warmups"]:
                    prefill.append(elapsed)
                    prefill_peaks.append(peak)
                    split_counts.append(numerical_split_count() - before)
                del logits, cache
            for iteration in range(repetitions):
                logits, cache = runtime.prefill(inputs[:, :length])
                del logits
                torch.npu.synchronize()
                torch.npu.reset_peak_memory_stats()
                start = time.perf_counter()
                for offset in range(protocol["decode_tokens"]):
                    logits, cache = runtime.step(inputs[:, length + offset], cache)
                torch.npu.synchronize()
                elapsed = time.perf_counter() - start
                peak = torch.npu.max_memory_allocated()
                after_bytes = cache.tensor_bytes()
                assert cache.position == length + protocol["decode_tokens"]
                assert torch.isfinite(logits).all().item()
                if iteration >= protocol["warmups"]:
                    decode.append(elapsed)
                    decode_peaks.append(peak)
                del logits, cache
            record = {"length": length, "batch": args.batch, "prefill": summarize(prefill, args.batch * length),
                      "decode": summarize(decode, args.batch * protocol["decode_tokens"]),
                      "decode_milliseconds_per_step": statistics.median(decode) / protocol["decode_tokens"] * 1000,
                      "cache_bytes_at_context": cache_bytes, "cache_bytes_after_decode": after_bytes,
                      "prefill_peak_allocated_bytes": max(prefill_peaks), "decode_peak_allocated_bytes": max(decode_peaks),
                      "reserved_bytes_after_case": torch.npu.memory_reserved(), "prefill_numerical_splits_per_repeat": split_counts}
            results.append(record)
            guard_exclusive()
            print(json.dumps(record), flush=True)
    result = {"binding": binding, "budget": model.parameter_budget(), "results": results,
              "torch": str(torch.__version__), "torch_npu": str(torch_npu.__version__),
              "ascend_visible_devices": os.environ.get("ASCEND_VISIBLE_DEVICES"), "npu_snapshot": snapshot,
              "scope": protocol["scope"]}
    temporary = target.with_suffix(".json.partial")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)


if __name__ == "__main__":
    main()
