"""Same-card, fixed-weight full-model forward/backward backend comparison."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch
import torch_npu
import fla
from ad_kernel.model import LinearLanguageModel, ModelConfig

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--device", default="npu:1")
ap.add_argument("--output", default="backend_benchmark.json")
ap.add_argument("--checkpoint", type=Path)
ap.add_argument("--input", type=Path)
args = ap.parse_args()
torch.set_num_threads(4)
torch.npu.set_device(args.device)
results = []
checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True) if args.checkpoint else None
cases = [(checkpoint["model_config"]["method"], checkpoint["model_config"]["feature_dim"])] if checkpoint else [("ad", 64), ("hh_exp", 384)]
for method, m in cases:
    for round_index in range(2):
        for backend in (["torch", "ascend-hybrid"] if round_index == 0 else ["ascend-hybrid", "torch"]):
            torch.manual_seed(11)
            model = LinearLanguageModel(ModelConfig(method=method, feature_dim=m, backend=backend)).to(args.device)
            if checkpoint:
                model.load_state_dict(checkpoint["model"])
            if args.input:
                batch = torch.load(args.input, map_location="cpu", weights_only=True)
                ids = torch.cat([batch["x"], batch["y"][:, -1:]], dim=1).to(args.device)
            else:
                ids = torch.randint(50257, (16, 1025), generator=torch.Generator().manual_seed(172)).to(args.device)
            torch.npu.reset_peak_memory_stats()
            timings = []
            for iteration in range(10):
                model.zero_grad(set_to_none=True)
                torch.npu.synchronize()
                start = time.perf_counter()
                with torch.autocast("npu", dtype=torch.bfloat16):
                    loss = model(ids[:, :-1], ids[:, 1:])
                loss.backward()
                torch.npu.synchronize()
                elapsed = time.perf_counter() - start
                assert torch.isfinite(loss).item()
                assert all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
                if iteration >= 3:
                    timings.append(elapsed)
            row = {"method": method, "m": m, "backend": backend, "round": round_index,
                   "median_seconds": statistics.median(timings), "timings_seconds": timings,
                   "tokens_per_second": 16384 / statistics.median(timings),
                   "peak_allocated_gib": torch.npu.max_memory_allocated() / 2**30}
            results.append(row)
            print(json.dumps(row), flush=True)
            del model, ids, loss
            gc.collect()
            torch.npu.empty_cache()
path = ROOT / "work/reproduction" / args.output
path.write_text(json.dumps({"device": args.device, "batch": 16, "length": 1024, "results": results,
                           "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                           "input": str(args.input) if args.input else None,
                           "scope": "Fixed weights, forward+backward only, fused CE common, 3 warmups + 7 samples, reversed backend order across rounds; no optimizer or data loading"}, indent=2) + "\n")
