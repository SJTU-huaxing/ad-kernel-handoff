"""Rebuild post-RoPE teacher activations from the frozen public manifest.

Workers own disjoint shards. Existing shards are reused only after checking their
manifest binding, SHA256, IDs, positions, shapes and finite activations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import torch_npu
from transformers import AutoModelForCausalLM
from transformers.models.qwen2 import modeling_qwen2

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate(path, meta, rows, manifest_hash):
    assert meta["manifest_sha256"] == manifest_hash
    assert meta["sha256"] == sha256(path)
    data = torch.load(path, map_location="cpu", weights_only=True)
    b, t, n = len(rows), len(rows[0]["input_ids"]), len(rows[0]["query_positions"])
    shapes = {"q": (b, 24, n, 128), "k": (b, 4, t, 128),
              "input_ids": (b, t), "query_positions": (b, n)}
    if rows[0]["split"] != "train":
        shapes["v"] = (b, 4, t, 128)
    assert set(data) == set(shapes)
    for key, shape in shapes.items():
        assert tuple(data[key].shape) == shape, (path, key, data[key].shape, shape)
        assert torch.isfinite(data[key]).all()
    for key in ("input_ids", "query_positions"):
        assert torch.equal(data[key], torch.tensor([r[key] for r in rows]))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(4)
    device = f"npu:{args.rank}"
    torch.npu.set_device(device)
    target = ROOT / "work/reproduction/public_wikitext_v1/data"
    manifest_path = target / "manifest.json"
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    protocol = json.loads((ROOT / "configs/reproduction_public_v1.json").read_text())
    records = manifest["records"]
    tasks = []
    for split in protocol["splits"]:
        rows = [r for r in records if r["split"] == split]
        for begin in range(0, len(rows), protocol["shard_documents"]):
            tasks.append((split, begin // protocol["shard_documents"], rows[begin:begin + protocol["shard_documents"]]))
    tasks = tasks[args.rank::args.workers]
    pending = []
    for split, index, rows in tasks:
        path = target / f"{split}_{index:03d}.pt"
        sidecar = path.with_suffix(".json")
        if path.exists() and sidecar.exists():
            validate(path, json.loads(sidecar.read_text()), rows, manifest_hash)
            print(json.dumps({"event": "verified_existing", "path": str(path)}), flush=True)
        elif path.exists():
            raise RuntimeError(f"Shard without audit sidecar: {path}")
        else:
            pending.append((split, index, rows))
    if not pending:
        print(json.dumps({"event": "complete", "rank": args.rank, "shards": len(tasks)}), flush=True)
        return
    model = AutoModelForCausalLM.from_pretrained(
        ROOT / "work/models/Qwen2.5-1.5B", local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    original = modeling_qwen2.ALL_ATTENTION_FUNCTIONS["sdpa"]
    captured = {}

    def capture(module, q, k, v, mask, **kwargs):
        if module.layer_idx in protocol["layers"]:
            captured[module.layer_idx] = (q, k, v)
        return original(module, q, k, v, mask, **kwargs)

    modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register("sdpa", capture)
    try:
        model.model(input_ids=torch.tensor([pending[0][2][0]["input_ids"]], device=device), use_cache=False)
        start = time.perf_counter()
        documents = tokens = 0
        for split, index, rows in pending:
            path = target / f"{split}_{index:03d}.pt"
            sidecar = path.with_suffix(".json")
            part = {key: [] for key in ["q", "k", "input_ids", "query_positions"]}
            if split != "train":
                part["v"] = []
            batch_size = args.batch_size if len(rows[0]["input_ids"]) <= 1024 else 1
            for begin in range(0, len(rows), batch_size):
                batch = rows[begin:begin + batch_size]
                ids = torch.tensor([r["input_ids"] for r in batch], device=device)
                positions = torch.tensor([r["query_positions"] for r in batch], device=device)
                model.model(input_ids=ids, use_cache=False)
                q = torch.cat([captured[layer][0] for layer in protocol["layers"]], dim=1)
                qi = positions[:, None, :, None].expand(-1, 24, -1, 128)
                part["q"].append(q.gather(2, qi).cpu())
                part["k"].append(torch.cat([captured[layer][1] for layer in protocol["layers"]], dim=1).cpu())
                if "v" in part:
                    part["v"].append(torch.cat([captured[layer][2] for layer in protocol["layers"]], dim=1).cpu())
                part["input_ids"].append(ids.cpu())
                part["query_positions"].append(positions.cpu())
                documents += len(batch)
                tokens += ids.numel()
            data = {key: torch.cat(value) for key, value in part.items()}
            temporary = path.with_suffix(".pt.partial")
            torch.save(data, temporary)
            meta = {"manifest_sha256": manifest_hash, "sha256": sha256(temporary),
                    "teacher_revision": protocol["model_revision"], "capture": "post-RoPE, pre-GQA-repeat",
                    "dtype": "bfloat16", "device": device, "batch_size": batch_size,
                    "documents": len(rows), "split": split, "shard": index,
                    "torch": torch.__version__, "torch_npu": torch_npu.__version__}
            validate(temporary, meta, rows, manifest_hash)
            temporary.replace(path)
            meta_temp = sidecar.with_suffix(".json.partial")
            meta_temp.write_text(json.dumps(meta, indent=2) + "\n")
            meta_temp.replace(sidecar)
            print(json.dumps({"event": "shard", "rank": args.rank, "path": str(path),
                              "documents": documents, "tokens": tokens,
                              "seconds": time.perf_counter() - start}), flush=True)
        print(json.dumps({"event": "complete", "rank": args.rank, "shards": len(tasks),
                          "seconds": time.perf_counter() - start}), flush=True)
    finally:
        modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register("sdpa", original)


if __name__ == "__main__":
    main()
