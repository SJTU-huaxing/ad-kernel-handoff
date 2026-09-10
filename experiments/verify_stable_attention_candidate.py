import json
import argparse
import hashlib
from pathlib import Path
import torch
import torch_npu
import fla
from ad_kernel.attention import dense_attention, chunk_attention, numerical_split_count
from stable_attention_candidate import stable_chunk

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/reproduction/training_failure_step348"
ap = argparse.ArgumentParser()
ap.add_argument("--integrated", action="store_true")
args = ap.parse_args()
torch.set_num_threads(8)
torch.npu.set_device("npu:0")
case = torch.load(OUT / "regression_layer7_b4_h6.pt", map_location="cpu", weights_only=True)
cases = {"actual_step348": [case[k] for k in ["logq", "logk", "v", "dy"]]}
for amplitude in [100., 200., 1000.]:
    torch.manual_seed(991)
    q, k = [torch.randn(1, 2, 129, 8) * 3 for _ in range(2)]
    k[:, :, 64:] += amplitude
    v, dy = [torch.randn(1, 2, 129, 6) for _ in range(2)]
    cases[f"future_amplitude_{int(amplitude)}"] = [q, k, v, dy]
records = []
for name, case in cases.items():
    x = [v.double().requires_grad_() for v in case[:3]]
    target = dense_attention(*x)
    grad = torch.autograd.grad(target, x, case[3].double())
    for device in ["cpu", "npu:0"]:
        other = [v.to(device).requires_grad_() for v in case[:3]]
        splits = []
        before = numerical_split_count()
        actual, _ = chunk_attention(*other) if args.integrated else stable_chunk(*other, stats=splits)
        measured = torch.autograd.grad(actual, other, case[3].to(device))
        errors = [float((a.detach().cpu().double() - b).square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-30)) for a, b in zip(measured, grad)]
        record = {"case": name, "device": device, "splits": splits, "integrated_split_count": numerical_split_count() - before,
                  "output_relative_rms": float((actual.detach().cpu().double() - target).square().mean().sqrt() / target.square().mean().sqrt()),
                  "output_max_absolute": float((actual.detach().cpu().double() - target).abs().max()),
                  "gradient_relative_rms": errors, "finite": all(torch.isfinite(g).all().item() for g in measured)}
        print(json.dumps(record), flush=True)
        assert record["finite"] and max(errors) < 2e-3 and record["output_relative_rms"] < 1e-4, record
        records.append(record)
(OUT / ("stable_integrated.json" if args.integrated else "stable_candidate.json")).write_text(json.dumps({"passed": True, "records": records,
    "attention_source_sha256": hashlib.sha256((ROOT / "src/ad_kernel/attention.py").read_bytes()).hexdigest()}, indent=2) + "\n")
