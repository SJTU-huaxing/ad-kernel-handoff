"""Hash-bound fixed-checkpoint quality evaluation, manually sharded across NPUs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from ad_kernel.model import LinearLanguageModel, ModelConfig
try:
    from .inference_runtime import LMRuntime
except ImportError:
    from inference_runtime import LMRuntime

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "work/pretraining/data/fineweb_edu_v1"
LONG = ROOT / "work/pretraining/evaluation/long_v1"
EXTRA = ROOT / "work/pretraining/evaluation/transfer_recall_v1"
PROTOCOL = ROOT / "configs/pretrained_evaluation_v1.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@torch.inference_mode()
def hidden_states(model, tokens):
    # The exact frozen forward blocks, without allocating full-vocabulary
    # logits for all sequence positions at once.
    x = model.embedding(tokens)
    angles = torch.arange(tokens.shape[1], device=tokens.device).float()[:, None] * model.inverse_frequency
    phase = torch.cat([angles, angles], -1)[None, None]
    cos, sin = phase.cos(), phase.sin()
    for block in model.blocks:
        x = block(x, cos, sin)
    return model.final_norm(x)


@torch.inference_mode()
def token_nll(model, tokens, targets):
    hidden = hidden_states(model, tokens).flatten(0, 1)
    labels = targets.flatten()
    parts = []
    for begin in range(0, len(hidden), 256):
        logits = F.linear(hidden[begin:begin + 256], model.embedding.weight).float()
        loss = F.cross_entropy(logits, labels[begin:begin + 256], reduction="none")
        parts.append(loss.cpu().double())
    return torch.cat(parts).reshape(targets.shape)


def totals(records):
    count = sum(row["tokens"] for row in records)
    loss = sum(row["nll_sum"] for row in records)
    nll = loss / count if count else None
    return {"tokens": count, "nll_sum": loss, "nll_per_token": nll,
            "perplexity": math.exp(nll) if nll is not None else None}


def ids(array, device):
    return torch.from_numpy(np.array(array, dtype=np.int64, copy=True)).to(device)


@torch.inference_mode()
def packed(model, device, rank, world, micro):
    stream = np.memmap(DATA / "test.bin", dtype="<u2", mode="r")
    indices = list(range(rank, (len(stream) - 1) // 1024, world))
    records = []
    for begin in range(0, len(indices), micro):
        subset = indices[begin:begin + micro]
        tokens = ids(np.stack([stream[i * 1024:i * 1024 + 1025] for i in subset]), device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            losses = token_nll(model, tokens[:, :-1], tokens[:, 1:])
        for i, loss in zip(subset, losses):
            records.append({"index": i, "tokens": 1024, "nll_sum": loss.sum().item()})
    return {"records": records, "summary": totals(records), "scope": "Complete fixed-length packed test blocks; target tokens do not overlap"}


@torch.inference_mode()
def long_documents(model, device, rank, world):
    arrays = np.load(LONG / "tokens.npy", mmap_mode="r", allow_pickle=False)
    records = []
    for index in range(rank, len(arrays), world):
        for length in [1024, 4096, 8192]:
            tokens = ids(arrays[index:index + 1, :length + 1], device)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                losses = token_nll(model, tokens[:, :-1], tokens[:, 1:])[0]
            row = {"index": index, "length": length, "tokens": length, "nll_sum": losses.sum().item()}
            if length == 8192:
                row["position_bands"] = [{"start": a, "end": b, "tokens": b - a, "nll_sum": losses[a:b].sum().item()}
                                          for a, b in [(0, 1024), (1024, 4096), (4096, 8192)]]
            records.append(row)
        print(json.dumps({"event": "long_document", "rank": rank, "index": index}), flush=True)
    return {"records": records, "summary_by_length": {str(length): totals([r for r in records if r["length"] == length])
                                                       for length in [1024, 4096, 8192]}}


@torch.inference_mode()
def wiki(model, device, rank, world):
    stream = np.memmap(EXTRA / "wiki.bin", dtype="<u2", mode="r")
    documents = json.loads((EXTRA / "wiki.documents.json").read_text())
    records = []
    for index in range(rank, len(documents), world):
        doc = documents[index]
        array = stream[doc["offset"]:doc["offset"] + doc["tokens"]]
        nll = 0.
        for begin in range(0, len(array) - 1, 1024):
            tokens = ids(array[begin:begin + 1025][None], device)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                nll += token_nll(model, tokens[:, :-1], tokens[:, 1:]).sum().item()
        records.append({"index": index, "source_row": doc["source_row"], "tokens": len(array) - 1, "nll_sum": nll})
    return {"records": records, "summary": totals(records), "scope": "WikiText raw test; document and 1024-target-block resets, final partial blocks retained"}


@torch.inference_mode()
def recall(model, device, rank, world):
    arrays = np.load(EXTRA / "recall.npz", allow_pickle=False)
    cases = json.loads((EXTRA / "recall.cases.json").read_text())
    records = []
    for index in range(rank, len(cases), world):
        case = cases[index]
        for length in [1024, 4096, 8192]:
            tokens = ids(arrays[str(length)][index:index + 1], device)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                hidden = hidden_states(model, tokens)[:, -1]
                logits = F.linear(hidden, model.embedding.weight).float()[0]
            log_probs = logits.log_softmax(-1).cpu()
            choices = log_probs[case["candidate_token_ids"]]
            gold = case["gold_candidate_index"]
            # Exact ties receive fractional credit; candidate ordering never
            # provides an arbitrary advantage to the first table entry.
            maxima = choices == choices.max()
            correct = float(maxima[gold]) / int(maxima.sum())
            rank_of_gold = 1 + int((choices > choices[gold]).sum()) + .5 * (int((choices == choices[gold]).sum()) - 1)
            records.append({"index": index, "length": length, "pairs": case["pairs"],
                            "target_position_group": case["target_position_group"], "candidate_accuracy": correct,
                            "gold_log_probability": float(log_probs[case["gold_token_id"]]), "gold_candidate_rank": rank_of_gold,
                            "unrestricted_correct": int(log_probs.argmax().item() == case["gold_token_id"]), "chance": 1 / case["pairs"]})
        print(json.dumps({"event": "recall_case", "rank": rank, "index": index}), flush=True)
    return {"records": records}


def logit_comparison(actual, expected):
    a, e = actual.float().cpu(), expected.float().cpu()
    error = a - e
    relative = error.square().mean().sqrt() / e.square().mean().sqrt().clamp_min(1e-12)
    kl = (e.softmax(-1) * (e.log_softmax(-1) - a.log_softmax(-1))).sum(-1).mean()
    result = {"relative_rms": float(relative), "max_absolute": error.abs().max().item(),
              "p99_absolute": torch.quantile(error.abs().flatten(), .99).item(), "mean_output_kl": kl.item()}
    assert result["relative_rms"] < .02 and result["p99_absolute"] <= .125 and result["max_absolute"] <= .5 and result["mean_output_kl"] < .005, result
    return result


@torch.inference_mode()
def cache_check(model, device):
    stream = np.memmap(DATA / "test.bin", dtype="<u2", mode="r")
    tokens = ids(stream[:1025][None], device)
    runtime = LMRuntime(model)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        expected = model(tokens[:, :129])
        first, cache = runtime.prefill(tokens[:, :63], all_logits=True)
        bytes63 = cache.tensor_bytes()
        middle, cache = runtime.prefill(tokens[:, 63:128], cache, all_logits=True)
        last, cache = runtime.step(tokens[:, 128], cache)
        short = logit_comparison(torch.cat([first, middle, last[:, None]], 1), expected)
        _, cache = runtime.prefill(tokens[:, :1024])
        bytes1024 = cache.tensor_bytes()
        final, cache = runtime.step(tokens[:, 1024], cache)
        full = F.linear(hidden_states(model, tokens)[:, -1], model.embedding.weight)
        long = logit_comparison(final, full)
    config = model.config
    if config.method == "softmax":
        # BF16 K/V, two arrays, all layers and heads.
        expected_bytes = config.layers * config.heads * (config.width // config.heads) * 1024 * 2 * 2
        assert bytes1024 == expected_bytes and cache.tensor_bytes() > bytes1024
    else:
        expected_bytes = model.parameter_budget()["linear_state_scalars_per_sequence"] * 4
        assert bytes63 == bytes1024 == cache.tensor_bytes() == expected_bytes
    return {"passed": True, "continued_129": short, "decode_1025": long,
            "bytes_at_63": bytes63, "bytes_at_1024": bytes1024, "bytes_at_1025": cache.tensor_bytes(),
            "tolerance_scope": "BF16 model-logit consistency; separate CPU FP64 attention/gradient and FP32 NPU tests enforce tighter kernel-level tolerances"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--stages", nargs="+", choices=["packed", "long", "wiki", "recall", "cache"], default=["cache", "packed", "long", "wiki", "recall"])
    args = ap.parse_args()
    assert 0 <= args.rank < args.world
    if args.device.startswith("npu"):
        import torch_npu
        import fla
        torch.npu.set_device(args.device)
    device = torch.device(args.device)
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    training = checkpoint["binding"]
    assert not training["synthetic"]
    assert training["protocol_sha256"] == digest(ROOT / "configs/pretraining_protocol_v2.json")
    assert training["data_manifest_sha256"] == digest(DATA / "manifest.json")
    for name, expected in training["code_sha256"].items():
        assert digest(ROOT / name) == expected, name
    assert checkpoint["step"] in [1600, 16384]
    assert checkpoint["tokens"] == checkpoint["step"] * 65536
    extra = json.loads((EXTRA / "manifest.json").read_text())
    long = json.loads((LONG / "manifest.json").read_text())
    data = json.loads((DATA / "manifest.json").read_text())
    assert extra["protocol_sha256"] == digest(PROTOCOL)
    assert long["protocol_sha256"] == digest(ROOT / "configs/long_context_evaluation_v1.json")
    assert extra["training_manifest_sha256"] == long["training_manifest_sha256"] == training["data_manifest_sha256"]
    assert digest(DATA / "test.bin") == data["artifacts"]["test.bin"]["sha256"]
    assert digest(LONG / "tokens.npy") == long["token_file_sha256"]
    for name, expected in extra["artifacts"].items():
        assert digest(EXTRA / name) == expected
    sources = [Path(__file__), ROOT / "experiments/inference_runtime.py", PROTOCOL]
    code_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    code_id = hashlib.sha256(json.dumps(code_hashes, sort_keys=True).encode()).hexdigest()[:12]
    binding = {"checkpoint_sha256": digest(args.checkpoint), "training": training, "step": checkpoint["step"],
               "tokens": checkpoint["tokens"], "evaluation_code_sha256": code_hashes,
               "long_manifest_sha256": digest(LONG / "manifest.json"), "extra_manifest_sha256": digest(EXTRA / "manifest.json"),
               "rank": args.rank, "world": args.world, "micro_batch": args.micro_batch, "device_type": device.type}
    out = ROOT / "work/pretraining/results" / f"{training['method']}_s{training['seed']}_step{checkpoint['step']:06d}" / code_id
    out.mkdir(parents=True, exist_ok=True)
    model = LinearLanguageModel(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.to(device).eval()
    for stage in args.stages:
        if stage == "cache" and args.rank != 0:
            continue
        path = out / f"{stage}.rank{args.rank}of{args.world}.json"
        if path.exists():
            assert json.loads(path.read_text())["binding"] == binding
            print(json.dumps({"event": "verified_skip", "stage": stage, "rank": args.rank}), flush=True)
            continue
        start = time.perf_counter()
        if stage == "packed":
            result = packed(model, device, args.rank, args.world, args.micro_batch)
        elif stage == "long":
            result = long_documents(model, device, args.rank, args.world)
        elif stage == "wiki":
            result = wiki(model, device, args.rank, args.world)
        elif stage == "recall":
            result = recall(model, device, args.rank, args.world)
        else:
            result = cache_check(model, device)
        save_json(path, {"binding": binding, "stage": stage, "result": result,
                         "wall_seconds": time.perf_counter() - start, "timing_scope": "Quality run; possibly concurrent with training, not an efficiency benchmark"})
        print(json.dumps({"event": "stage_complete", "stage": stage, "rank": args.rank, "path": str(path)}), flush=True)


if __name__ == "__main__":
    main()
