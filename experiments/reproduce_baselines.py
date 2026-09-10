"""Historical independent-Q/K Hedgehog adapters and fixed FAVOR, new data."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
import torch_npu
from ad_kernel.baselines import HedgehogFeatures, FavorFeatures
from reproduce_features import ROOT, RUN, KI, data, digest, evaluate, rows_kl, save_json, save_torch


def fit(kind, seed, lr, device, ds, val, stats):
    name = f"{kind}_s{seed}" + (f"_lr{lr:g}" if kind != "favor" else "")
    path = RUN / "fits" / f"{name}.pt"
    report = path.with_suffix(".json")
    binding = {"manifest_sha256": digest(RUN / "data/manifest.json"),
               "protocol_sha256": digest(ROOT / "configs/reproduction_public_v1.json"),
               "trainer_sha256": digest(Path(__file__)),
               "features_sha256": digest(ROOT / "src/ad_kernel/baselines.py")}
    if report.exists():
        old = json.loads(report.read_text())
        assert old["binding"] == binding and old["checkpoint_sha256"] == digest(path)
        return old
    torch.manual_seed(seed)
    net = (FavorFeatures(norm=stats["norm"]) if kind == "favor" else
           HedgehogFeatures(norm=stats["norm"], softmax=kind == "hh_softmax")).to(device)
    expected_parameters = 0 if kind == "favor" else 73728
    assert net.parameters_per_head == expected_parameters
    history, elapsed = [], 0.
    order = torch.randperm(4096, generator=torch.Generator().manual_seed(85000 + seed)).tolist()
    if kind != "favor":
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 4096, eta_min=lr / 10)
        index, keys, scale = KI.to(device), torch.arange(1024, device=device), stats["scale"].to(device)
        running, start = torch.zeros(24, device=device), time.perf_counter()
        for step, i in enumerate(order, 1):
            q, k = ds["q"][i].float(), ds["k"][i].float()[index]
            lt = q @ k.transpose(-1, -2) / math.sqrt(128) - scale[:, None, None]
            lp = net.log_matrix(q, k)
            mask = (keys[None] <= ds["query_positions"][i, :, None])[None]
            losses = rows_kl(lp, lt, mask).mean(-1)
            optimizer.zero_grad(set_to_none=True)
            losses.mean().backward()
            sums = sum(parameter.grad.flatten(1).square().sum(-1) for parameter in net.parameters())
            factor = (10 / sums.sqrt().clamp_min(1e-20)).clamp_max(1)
            for parameter in net.parameters():
                parameter.grad.mul_(factor.reshape(24, *([1] * (parameter.ndim - 1))))
            optimizer.step()
            scheduler.step()
            running += losses.detach()
            if step % 512 == 0:
                perhead = (running / 512).tolist()
                assert all(math.isfinite(x) for x in perhead)
                row = {"step": step, "per_head_loss": perhead, "seconds": time.perf_counter() - start}
                history.append(row)
                print(json.dumps({"event": "train", "name": name, "step": step,
                                  "loss": sum(perhead) / 24, "seconds": row["seconds"]}), flush=True)
                running.zero_()
        torch.npu.synchronize()
        elapsed = time.perf_counter() - start
    meta = {"name": name, "kind": kind, "seed": seed, "lr": lr, "m": net.m,
            "parameters_per_head": expected_parameters, "training_seconds": elapsed,
            "training_steps": 0 if kind == "favor" else 4096,
            "train_documents": 0 if kind == "favor" else 4096,
            "training_pairs_per_head": 0 if kind == "favor" else int((ds["query_positions"] + 1).sum().item()),
            "order_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest(),
            "scope": "Historical feature adapter on new public QKV; not a full paper reproduction",
            "device": device, "torch": torch.__version__, "torch_npu": torch_npu.__version__}
    save_torch(path, {"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "metadata": meta, "binding": binding})
    metric = evaluate(net, val)
    score = sum(metric["summary"]["kl"]) / 24
    result = {"metadata": meta, "binding": binding, "checkpoint_sha256": digest(path),
              "history": history, "validation": metric, "selection_score": score}
    save_json(report, result)
    print(json.dumps({"event": "fit", "name": name, "validation_kl": score}), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["hh_exp", "hh_softmax", "favor"], required=True)
    ap.add_argument("--device", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    assert json.loads((ROOT / "work/reproduction/baseline_port.json").read_text())["passed"]
    stats = torch.load(RUN / "normalization.pt", weights_only=True)
    assert stats["manifest_sha256"] == digest(RUN / "data/manifest.json")
    ds = None if args.kind == "favor" else {key: value.to(args.device) for key, value in data("train").items()}
    val = data("validation", limit=32)
    if args.kind == "favor":
        for seed in [11, 29, 47]:
            fit(args.kind, seed, 0., args.device, ds, val, stats)
        return
    trials = [fit(args.kind, 11, lr, args.device, ds, val, stats) for lr in [.002, .0005]]
    lr = min(trials, key=lambda x: x["selection_score"])["metadata"]["lr"]
    save_json(RUN / f"selection_{args.kind}.json", {"lr": lr, "kind": args.kind,
              "seed11_validation_scores": {str(t["metadata"]["lr"]): t["selection_score"] for t in trials},
              "manifest_sha256": digest(RUN / "data/manifest.json"), "criterion": "First 32 validation documents only"})
    for seed in [29, 47]:
        fit(args.kind, seed, lr, args.device, ds, val, stats)
    print(json.dumps({"event": "grid_complete", "kind": args.kind, "selected_lr": lr}), flush=True)


if __name__ == "__main__":
    main()
