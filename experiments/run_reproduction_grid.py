"""Run the preregistered AD/EXP LR selection and all three feature seeds."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from reproduce_features import RUN, digest, save_json

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--kind", choices=["ad", "exp"], required=True)
ap.add_argument("--device", required=True)
args = ap.parse_args()
selected = {}
for initialization in ["standard", "matched_zero"]:
    def run(seed, lr):
        subprocess.run([sys.executable, str(ROOT / "experiments/reproduce_features.py"), "train",
                        "--kind", args.kind, "--initialization", initialization,
                        "--seed", str(seed), "--lr", str(lr), "--device", args.device], check=True)
        path = RUN / "fits" / f"{initialization}_{args.kind}_s{seed}_lr{lr:g}.json"
        return json.loads(path.read_text())
    trials = [run(11, lr) for lr in [.002, .0005]]
    lr = min(trials, key=lambda trial: trial["selection_score"])["metadata"]["lr"]
    selected[initialization] = {"lr": lr, "seed11_validation_scores": {
        str(trial["metadata"]["lr"]): trial["selection_score"] for trial in trials}}
    save_json(RUN / f"selection_{args.kind}.json", {"kind": args.kind, "selected": selected,
              "manifest_sha256": digest(RUN / "data/manifest.json"),
              "criterion": "Seed 11 mean KL on first 32 validation documents; no confirmation metrics accessed"})
    for seed in [29, 47]:
        run(seed, lr)
print(json.dumps({"event": "grid_complete", "kind": args.kind, "selected": selected}), flush=True)
