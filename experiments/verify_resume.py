"""Compare resumed and uninterrupted distributed synthetic training."""
import argparse
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--resumed", default="resume-check-ad")
ap.add_argument("--reference", default="uninterrupted-check-ad")
ap.add_argument("--output", default="resume_check.json")
args = ap.parse_args()
torch.set_num_threads(4)
base = ROOT / "work/pretraining/runs"
a = torch.load(base / args.resumed / "last.pt", weights_only=True, map_location="cpu")
b = torch.load(base / args.reference / "last.pt", weights_only=True, map_location="cpu")
assert a["step"] == b["step"] == 4
assert a["tokens"] == b["tokens"] == 262144
assert a["binding"] == b["binding"] and a["order_sha256"] == b["order_sha256"]
assert a["data_cursor_sequences"] == b["data_cursor_sequences"] == 256
errors = {name: (value - b["model"][name]).abs().max().item() for name, value in a["model"].items()}
maximum = max(errors.values())
optimizer_max = 0.
for key, state in a["optimizer"]["state"].items():
    for field, value in state.items():
        expected = b["optimizer"]["state"][key][field]
        if isinstance(value, torch.Tensor):
            optimizer_max = max(optimizer_max, (value - expected).abs().max().item())
        else:
            assert value == expected
assert maximum < 1e-5, sorted(errors.items(), key=lambda pair: -pair[1])[:5]
assert optimizer_max < 1e-5, optimizer_max
for ra, rb in zip(a["rng_by_rank"], b["rng_by_rank"]):
    assert torch.equal(ra["torch"], rb["torch"]) and torch.equal(ra["npu"], rb["npu"])
    assert ra["python"] == rb["python"]
result = {"passed": True, "synthetic": True, "steps": 4, "tokens": 262144,
          "binding": a["binding"],
          "parameter_max_abs": maximum, "optimizer_state_max_abs": optimizer_max,
          "data_cursor_and_rng_match": True}
(ROOT / "work/reproduction" / args.output).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
