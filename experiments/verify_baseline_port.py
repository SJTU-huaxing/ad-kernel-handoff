"""Verify all nine historical HH/FAVOR checkpoints against original code."""
import importlib.util
import json
from pathlib import Path
import sys
import torch
from ad_kernel.baselines import HedgehogFeatures, FavorFeatures

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "work/kan_attention_theory"
sys.path.insert(0, str(SOURCE / "hedgehog_matched"))
spec = importlib.util.spec_from_file_location("original_hedgehog_models", SOURCE / "hedgehog_matched/models.py")
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
torch.set_num_threads(4)
results = []
for family in ["hh_kl", "hh_softmax_kl", "favor"]:
    for seed in [11, 29, 47]:
        filename = f"{family}_s{seed}" + ("_lr0.002" if family != "favor" else "") + ".pt"
        directory = "hedgehog_matched" if family != "favor" else "causal_direction"
        path = SOURCE / directory / "fits" / filename
        box = torch.load(path, weights_only=True, map_location="cpu")
        state = box["state_dict"]
        norm = {k: state[k] for k in ["q_mean", "q_std", "k_mean", "k_std"]}
        if family == "favor":
            source = original.source.Baseline(norm, "learned_prf").double()
            target = FavorFeatures(norm=norm).double()
        else:
            source = original.Matched(norm, family).double()
            target = HedgehogFeatures(norm=norm, softmax=family == "hh_softmax_kl").double()
        source.load_state_dict(state)
        target.load_state_dict(state)
        torch.manual_seed(seed + 907)
        q, k = [torch.randn(24, t, 128, dtype=torch.float64) for t in [7, 11]]
        expected, actual = source.log_matrix(q, k), target.log_matrix(q, k)
        error = (expected - actual).abs().max().item()
        assert error == 0, (filename, error)
        results.append({"checkpoint": str(path.relative_to(ROOT)), "log_matrix_max_abs": error,
                        "parameters_per_head": target.parameters_per_head, "m": target.m})
path = ROOT / "work/reproduction/baseline_port.json"
path.write_text(json.dumps({"passed": True, "results": results}, indent=2) + "\n")
print(json.dumps({"passed": True, "checkpoints": len(results), "path": str(path)}), flush=True)
