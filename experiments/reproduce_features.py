"""Same-depth AD/EXP fitting on frozen, newly extracted teacher QKV.

One document per optimizer step, original seed/order/schedule and per-head
gradient clipping. This is a new-data reproduction, not the missing GPU dataset.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
import torch_npu
from ad_kernel.features import ADFeatures

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "work/reproduction/public_wikitext_v1"
KI = torch.tensor([i // 6 for i in range(24)])


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def save_torch(path, data):
    temp = path.with_suffix(path.suffix + ".partial")
    torch.save(data, temp)
    temp.replace(path)


def data(split, limit=None):
    manifest_hash = digest(RUN / "data/manifest.json")
    rows, count = [], 0
    for path in sorted((RUN / "data").glob(f"{split}_*.pt")):
        audit = json.loads(path.with_suffix(".json").read_text())
        assert audit["manifest_sha256"] == manifest_hash and audit["sha256"] == digest(path)
        row = torch.load(path, weights_only=True, map_location="cpu")
        rows.append(row)
        count += len(row["q"])
        if limit is not None and count >= limit:
            break
    assert rows
    return {key: torch.cat([row[key] for row in rows])[:limit] for key in rows[0]}


def prepare_stats():
    destination = RUN / "normalization.pt"
    binding = digest(RUN / "data/manifest.json")
    if destination.exists():
        assert torch.load(destination, weights_only=True)["manifest_sha256"] == binding
        return
    ds = data("train")
    norm = {}
    for side in ["q", "k"]:
        x = ds[side][:, :, ::8].permute(1, 0, 2, 3).flatten(1, 2).double()
        mean, std = x.mean(1).float(), x.std(1).clamp_min(.03).float()
        if side == "k":
            mean, std = mean[KI], std[KI]
        norm[side + "_mean"], norm[side + "_std"] = mean, std
        del x
    logs = []
    for begin in range(0, len(ds["q"]), 32):
        q = ds["q"][begin:begin + 32].float()
        k = ds["k"][begin:begin + 32, :, ::16][:, KI].float()
        logs.append((q * k).sum(-1) / math.sqrt(128))
    x = torch.cat(logs).permute(1, 0, 2).flatten(1).double()
    scale = (x.logsumexp(-1) - math.log(x.shape[1])).float()
    save_torch(destination, {"norm": norm, "scale": scale, "manifest_sha256": binding,
                             "scope": "training only, original subsampling and unbiased std; CPU FP64 reduction"})
    print(json.dumps({"event": "normalization", "sha256": digest(destination)}), flush=True)


def rows_kl(lp, lt, mask):
    t, p = lt.masked_fill(~mask, -torch.inf), lp.masked_fill(~mask, -torch.inf)
    z, zh = t.logsumexp(-1), p.logsumexp(-1)
    a = (t - z[..., None]).exp()
    residual = (lt - z[..., None] - lp + zh[..., None]).masked_fill(~mask, 0)
    return (a * residual).sum(-1)


@torch.inference_mode()
def evaluate(net, ds):
    net = net.cpu().double().eval()
    rows = []
    for i in range(len(ds["q"])):
        q, k, v = ds["q"][i].double(), ds["k"][i, KI].double(), ds["v"][i, KI].double()
        mask = (torch.arange(k.shape[1])[None] <= ds["query_positions"][i, :, None])[None]
        lt, lp = q @ k.transpose(-1, -2) / math.sqrt(128), net.log_matrix(q, k)
        a, b = lt.masked_fill(~mask, -torch.inf).softmax(-1), lp.masked_fill(~mask, -torch.inf).softmax(-1)
        y, yh = a @ v, b @ v
        row = {"ordinal": i, "kl": rows_kl(lp, lt, mask).mean(-1).tolist(),
               "tv": ((a - b).abs().sum(-1) / 2).mean(-1).tolist(),
               "coefficient_l2": (a - b).square().sum(-1).mean(-1).tolist(),
               "output_nmse": ((y - yh).square().sum((-1, -2)) / y.square().sum((-1, -2)).clamp_min(1e-30)).tolist()}
        assert all(math.isfinite(value) for key in row if key != "ordinal" for value in row[key])
        rows.append(row)
    summary = {key: torch.tensor([r[key] for r in rows], dtype=torch.float64).mean(0).tolist()
               for key in rows[0] if key != "ordinal"}
    return {"dtype": "CPU float64", "summary": summary, "documents": rows}


def train(args):
    name = f"{args.initialization}_{args.kind}_s{args.seed}_lr{args.lr:g}"
    directory = RUN / "fits"
    directory.mkdir(exist_ok=True)
    path = directory / f"{name}.pt"
    report = path.with_suffix(".json")
    resume = directory / f"{name}.resume.pt"
    binding = {"manifest_sha256": digest(RUN / "data/manifest.json"),
               "protocol_sha256": digest(ROOT / "configs/reproduction_public_v1.json"),
               "trainer_sha256": digest(Path(__file__)),
               "feature_source_sha256": digest(ROOT / "src/ad_kernel/features.py")}
    if report.exists():
        old = json.loads(report.read_text())
        assert old["binding"] == binding and old["checkpoint_sha256"] == digest(path)
        print(json.dumps({"event": "verified_existing_fit", "name": name}), flush=True)
        return
    if path.exists() and not resume.exists():
        raise RuntimeError(f"Completed weights lack report; evaluate explicitly: {path}")
    stats = torch.load(RUN / "normalization.pt", weights_only=True)
    assert stats["manifest_sha256"] == binding["manifest_sha256"]
    device = args.device
    torch.npu.set_device(device)
    ds = {key: value.to(device) for key, value in data("train").items()}
    val = data("validation", limit=32)
    torch.manual_seed(args.seed)
    net = ADFeatures(kind=args.kind, initialization=args.initialization,
                     norm=stats["norm"], numerical_scale=stats["scale"]).to(device)
    net.runtime_raw = False  # Same numerical representation as historical fitting.
    n = len(ds["q"])
    assert n == 4096 and net.parameters_per_head == 73536
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n, eta_min=args.lr / 10)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(85000 + args.seed)).tolist()
    index, keys = KI.to(device), torch.arange(1024, device=device)
    history, completed, previous_seconds = [], 0, 0.
    if resume.exists():
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        assert checkpoint["binding"] == binding
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history, completed, previous_seconds = checkpoint["history"], checkpoint["step"], checkpoint["seconds"]
        torch.set_rng_state(checkpoint["rng"])
        torch.npu.set_rng_state(checkpoint["npu_rng"], device=device)
    start = time.perf_counter()
    running = torch.zeros(24, device=device)
    print(json.dumps({"event": "start", "name": name, "device": device, "completed": completed}), flush=True)
    for step in range(completed + 1, n + 1):
        i = order[step - 1]
        q, k = ds["q"][i].float(), ds["k"][i].float()[index]
        lt = q @ k.transpose(-1, -2) / math.sqrt(128) - net.numerical_scale[:, None, None]
        lp = net.log_matrix(q, k)
        mask = (keys[None] <= ds["query_positions"][i, :, None])[None]
        perhead = rows_kl(lp, lt, mask).mean(-1)
        optimizer.zero_grad(set_to_none=True)
        perhead.mean().backward()
        sums = torch.zeros(24, device=device)
        for parameter in net.parameters():
            sums += parameter.grad.flatten(1).square().sum(-1)
        factor = (10 / sums.sqrt().clamp_min(1e-20)).clamp_max(1)
        for parameter in net.parameters():
            parameter.grad.mul_(factor.reshape(24, *([1] * (parameter.ndim - 1))))
        optimizer.step()
        scheduler.step()
        running += perhead.detach()
        if step % 256 == 0:
            losses = (running / 256).tolist()
            assert all(math.isfinite(x) for x in losses), (name, step)
            row = {"step": step, "per_head_loss": losses,
                   "seconds": previous_seconds + time.perf_counter() - start}
            history.append(row)
            running.zero_()
            print(json.dumps({"event": "train", "name": name, "step": step,
                              "loss": sum(losses) / 24, "seconds": row["seconds"]}), flush=True)
        if step % 1024 == 0:
            save_torch(resume, {"binding": binding, "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                               "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                               "rng": torch.get_rng_state(), "npu_rng": torch.npu.get_rng_state(device),
                               "history": history, "step": step,
                               "seconds": previous_seconds + time.perf_counter() - start})
    torch.npu.synchronize()
    meta = {"name": name, "map_kind": args.kind, "initialization": args.initialization,
            "seed": args.seed, "lr": args.lr, "m": 64, "parameters_per_head": 73536,
            "train_documents": n, "training_queries_per_head": n * 64, "epochs": 1,
            "training_pairs_per_head": int((ds["query_positions"] + 1).sum().item()),
            "order_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest(),
            "training_seconds": previous_seconds + time.perf_counter() - start,
            "device": device, "torch": torch.__version__, "torch_npu": torch_npu.__version__,
            "scope": "New public-data reproduction, original GPU data unavailable"}
    save_torch(path, {"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "metadata": meta, "binding": binding})
    del ds
    metric = evaluate(net, val)
    score = sum(metric["summary"]["kl"]) / 24
    save_json(report, {"metadata": meta, "binding": binding, "checkpoint_sha256": digest(path),
                       "history": history, "validation": metric, "selection_score": score})
    print(json.dumps({"event": "fit", "name": name, "validation_kl": score}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["stats", "train"])
    ap.add_argument("--kind", choices=["ad", "exp"], default="ad")
    ap.add_argument("--initialization", choices=["standard", "matched_zero"], default="standard")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--lr", type=float, default=.002)
    ap.add_argument("--device", default="npu:0")
    args = ap.parse_args()
    torch.set_num_threads(4)
    prepare_stats() if args.action == "stats" else train(args)
