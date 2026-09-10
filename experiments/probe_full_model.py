"""Run a real-size model step; timing is diagnostic until cards are exclusive."""
import argparse
import json
from pathlib import Path
import time
import torch
import torch_npu
import fla
from ad_kernel.model import LinearLanguageModel, ModelConfig

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--method", default="ad")
ap.add_argument("--m", type=int, default=64)
ap.add_argument("--batch-size", type=int, default=2)
ap.add_argument("--length", type=int, default=1024)
ap.add_argument("--device", default="npu:0")
ap.add_argument("--steps", type=int, default=3)
ap.add_argument("--backend", choices=["torch", "ascend-hybrid"], default="ascend-hybrid")
args = ap.parse_args()
torch.set_num_threads(4)
torch.npu.set_device(args.device)
torch.manual_seed(11)
config = ModelConfig(method=args.method, feature_dim=args.m, backend=args.backend)
model = LinearLanguageModel(config).to(args.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(.9,.95), weight_decay=.1)
ids = torch.randint(config.vocab_size, (args.batch_size, args.length + 1), device=args.device)
print(json.dumps({"event": "start", "budget": model.parameter_budget(), "batch": args.batch_size}), flush=True)
steps = []
for step in range(args.steps):
    torch.npu.synchronize()
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("npu", dtype=torch.bfloat16):
        loss = model(ids[:, :-1], ids[:, 1:])
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all().item(), name
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    torch.npu.synchronize()
    row = {"step": step, "loss": loss.item(), "grad_norm": grad.item(),
           "seconds": time.perf_counter() - start,
           "peak_allocated_gib": torch.npu.max_memory_allocated() / 2**30}
    assert torch.isfinite(loss).item()
    steps.append(row)
    print(json.dumps(row), flush=True)
path = ROOT / f"work/reproduction/full_model_{args.method}_m{args.m}_b{args.batch_size}_{args.backend}.json"
path.write_text(json.dumps({"passed": True, "budget": model.parameter_budget(), "steps": steps,
                           "scope": "Synthetic numerical/size probe, concurrent work may affect timing"}, indent=2) + "\n")
