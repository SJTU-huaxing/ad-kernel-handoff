"""Supplementary same-target context and absolute-position controls."""
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

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "work/pretraining"
PROTOCOL = ROOT / "configs/context_control_v1.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


@torch.inference_mode()
def hidden_at_positions(model, tokens, offset=0):
    hidden = model.embedding(tokens)
    positions = torch.arange(offset, offset + tokens.shape[1], device=tokens.device).float()
    angles = positions[:, None] * model.inverse_frequency
    phase = torch.cat([angles, angles], -1)[None, None]
    cos, sin = phase.cos(), phase.sin()
    for block in model.blocks:
        hidden = block(hidden, cos, sin)
    return model.final_norm(hidden)


@torch.inference_mode()
def score_tail(model, tokens, targets, offset=0):
    assert tokens.shape[0] == targets.shape[0] and tokens.shape[1] >= targets.shape[1]
    hidden = hidden_at_positions(model, tokens, offset)[:, -targets.shape[1]:].flatten(0, 1)
    labels = targets.flatten()
    losses = []
    for start in range(0, len(hidden), 256):
        logits = F.linear(hidden[start:start + 256], model.embedding.weight).float()
        losses.append(F.cross_entropy(logits, labels[start:start + 256], reduction="none").cpu().double())
    return torch.cat(losses).reshape(targets.shape)


def context_and_targets(document, context_tokens, target_tokens=1024):
    assert len(document) == 8193 and target_tokens <= context_tokens <= 8192
    inputs = np.asarray(document[8192 - context_tokens:8192], dtype=np.int64)
    targets = np.asarray(document[8193 - target_tokens:8193], dtype=np.int64)
    assert len(inputs) == context_tokens and len(targets) == target_tokens
    return inputs, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--rank", type=int, choices=[0, 1], required=True)
    args = ap.parse_args()
    import torch_npu
    import fla
    device = f"npu:{args.rank}"
    torch.set_num_threads(4)
    torch.npu.set_device(device)
    config = json.loads(PROTOCOL.read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    training = checkpoint["binding"]
    assert not training["synthetic"] and training["seed"] == 11 and checkpoint["step"] == 16384
    assert checkpoint["tokens"] == 1073741824
    assert training["protocol_sha256"] == digest(ROOT / "configs/pretraining_protocol_v2.json")
    for source, expected in training["code_sha256"].items():
        assert digest(ROOT / source) == expected, source
    long_path = BASE / "evaluation/long_v1"
    manifest = json.loads((long_path / "manifest.json").read_text())
    assert digest(long_path / "tokens.npy") == manifest["token_file_sha256"]
    assert manifest["training_manifest_sha256"] == training["data_manifest_sha256"]
    code_hashes = {str(path.relative_to(ROOT)): digest(path) for path in [Path(__file__), PROTOCOL]}
    code_id = hashlib.sha256(json.dumps(code_hashes, sort_keys=True).encode()).hexdigest()[:12]
    binding = {"training": training, "checkpoint_sha256": digest(args.checkpoint), "source_sha256": code_hashes,
               "long_manifest_sha256": digest(long_path / "manifest.json"), "rank": args.rank, "world": 2,
               "step": checkpoint["step"], "tokens": checkpoint["tokens"]}
    out = BASE / "context_controls" / f"{training['method']}_s11_step016384" / code_id
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"rank{args.rank}.json"
    if path.exists():
        assert json.loads(path.read_text())["binding"] == binding
        print("Verified existing context-control shard", flush=True)
        return
    model = LinearLanguageModel(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.to(device).eval()
    data = np.load(long_path / "tokens.npy", mmap_mode="r", allow_pickle=False)
    records, start = [], time.perf_counter()
    for index in range(args.rank, len(data), 2):
        target_hash = None
        for condition in config["conditions"]:
            x, y = context_and_targets(data[index], condition["context_tokens"])
            current_hash = hashlib.sha256(y.astype("<i8").tobytes()).hexdigest()
            assert target_hash is None or current_hash == target_hash
            target_hash = current_hash
            x, y = [torch.from_numpy(array.copy()[None]).to(device) for array in [x, y]]
            with torch.autocast("npu", dtype=torch.bfloat16):
                loss = score_tail(model, x, y, condition["position_offset"])
            nll = loss.sum().item()
            assert math.isfinite(nll)
            records.append({"index": index, "condition": condition["name"], "tokens": 1024, "nll_sum": nll,
                            "target_sha256": target_hash, "context_tokens": condition["context_tokens"],
                            "position_offset": condition["position_offset"]})
        print(json.dumps({"event": "context_document", "rank": args.rank, "index": index}), flush=True)
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps({"binding": binding, "records": records, "seconds": time.perf_counter() - start}, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"event": "context_controls_complete", "path": str(path)}), flush=True)


if __name__ == "__main__":
    main()
