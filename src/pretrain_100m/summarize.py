"""Export observed training curves and final metrics; never invent missing runs."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from .common import ROOT, atomic_json


def main():
    base = ROOT/"work/pretrain_100m_2b"
    output = base/"reports"
    output.mkdir(parents=True, exist_ok=True)
    rows, curves = [], []
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for run in sorted((base/"runs").glob("*")):
        if not (run/"run.json").exists(): continue
        binding = json.loads((run/"run.json").read_text())["binding"]
        if binding["probe"]: continue
        # A restarted checkpoint can overlap earlier logged updates. Last write wins.
        logs = {x["step"]: x for x in [json.loads(line) for line in (run/"train.jsonl").read_text().splitlines()]} if (run/"train.jsonl").exists() else {}
        logs = [logs[k] for k in sorted(logs)]
        if logs:
            x, y = np.array([r["tokens"] for r in logs]), np.array([r["loss"] for r in logs])
            ax.plot(x/1e9, y, alpha=.15, linewidth=.5)
            window = min(50, len(logs))
            ax.plot(x[window-1:]/1e9, np.convolve(y, np.ones(window)/window, mode="valid"), label=run.name)
            curves += [{"method": binding["method"], "seed": binding["seed"], **r} for r in logs]
        summary = run/"evaluation/summary.json"
        if summary.exists():
            e = json.loads(summary.read_text())
            for stage, metrics in e["metrics"].items():
                if stage in ["validation", "test", "wiki"]:
                    rows.append({"method": binding["method"], "seed": binding["seed"], "stage": stage,
                                 "trained_tokens": e["tokens_trained"], **metrics})
    for name, records in [("metrics.csv", rows), ("training_loss.csv", curves)]:
        if records:
            with (output/name).open("w") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0]))
                writer.writeheader(); writer.writerows(records)
    if curves:
        ax.set(xlabel="Training target tokens (billions)", ylabel="Training cross entropy (nats/token)")
        ax.grid(alpha=.2); ax.legend()
        fig.savefig(output/"training_loss.png", dpi=180)
        fig.savefig(output/"training_loss.pdf")
    plt.close(fig)
    aggregate = []
    for method,stage in sorted(set((r["method"],r["stage"]) for r in rows)):
        group = [r for r in rows if r["method"] == method and r["stage"] == stage]
        aggregate.append({"method":method,"stage":stage,"seeds":[r["seed"] for r in group],
                          **{f"{metric}_{stat}":float(np.mean([r[metric] for r in group]) if stat == "mean" else np.std([r[metric] for r in group],ddof=1))
                             for metric in ["nll","ppl"] for stat in (["mean","std"] if len(group)>1 else ["mean"])}})
    atomic_json(output/"summary.json", {"final_metrics": rows, "seed_aggregates":aggregate,"logged_training_steps": len(curves),
                                       "scope": "Only completed evaluations; two-seed differences are descriptive, not a significance claim."})
    print(str(output))


if __name__ == "__main__": main()
