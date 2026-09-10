"""Exercise the repaired execution path on the actual failed full-model batch."""
import json
import argparse
import hashlib
from pathlib import Path
import torch
import torch_npu
import fla
import ad_kernel.model as model_module
from ad_kernel.model import LinearLanguageModel, ModelConfig
from stable_attention_candidate import stable_chunk

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/reproduction/training_failure_step348"
ap = argparse.ArgumentParser()
ap.add_argument("--integrated", action="store_true")
ap.add_argument("--device", default="npu:1")
args = ap.parse_args()
torch.set_num_threads(4)
torch.npu.set_device(args.device)
box = torch.load(OUT / "before_step348.pt", map_location="cpu", weights_only=True)
model = LinearLanguageModel(ModelConfig(**box["model_config"])).to(args.device)
model.load_state_dict(box["model"])
del box
case = torch.load(OUT / "batch_rank1_micro1.pt", map_location="cpu", weights_only=True)
splits = []
if not args.integrated:
    model_module.chunk_attention = lambda *args, **kwargs: stable_chunk(*args, stats=splits, **kwargs)
with torch.autocast("npu", dtype=torch.bfloat16):
    loss = model(case["x"].to(args.device), case["y"].to(args.device)) / 2
loss.backward()
bad = [name for name, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all().item()]
result = {"passed": not bad, "bad_parameters": bad, "loss": loss.item(), "splits": splits,
          "gradient_norm": torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item(),
          "attention_source_sha256": hashlib.sha256((ROOT / "src/ad_kernel/attention.py").read_bytes()).hexdigest()}
print(json.dumps(result), flush=True)
assert not bad
(OUT / ("full_model_integrated.json" if args.integrated else "full_model_candidate.json")).write_text(json.dumps(result, indent=2) + "\n")
