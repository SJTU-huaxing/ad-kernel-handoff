"""CPU-only export of the explicitly stopped seed 11/29 experiments.

Reads existing metrics and logs; never loads a model or starts a training job.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import shutil
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/pretraining/stopped_report"
ORDER = ["ad64", "exp64", "softmax", "hh_softmax128", "hh_exp128", "favor256", "favor64", "elu64", "softplus64", "hh_exp384"]
LABELS = dict(zip(ORDER, ["AD m64", "EXP m64", "Softmax", "HH-softmax m128", "HH-exp m128", "FAVOR+ m256", "FAVOR+ m64", "ELU+1 m64", "Softplus m64", "HH-exp m384"]))


def read(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|", *["| " + " | ".join(map(str, r)) + " |" for r in rows]])


def save_figure(fig, name):
    for extension in ["png", "svg"]:
        fig.savefig(OUT / "figures" / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    stop = read("work/pretraining/stopped_by_user.json")
    assert stop["remaining_live"] == []
    summary = read("work/pretraining/reports/summary.json")
    runs = sorted(summary["runs"], key=lambda r: (r["seed"], ORDER.index(r["method"])))
    assert len(runs) == 17 and {r["seed"] for r in runs} == {11, 29}
    assert all(r["step"] == 1600 and r["training_tokens"] == 104857600 for r in runs)
    costs = {(r["method"], r["seed"]): r for r in read("work/pretraining/reports/training_cost.json")["invocations"] if r["status"] == "complete"}
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    metrics, losses, recall, bands, caches = [], [], [], [], []
    logs, budgets, provenance = {}, {}, {}
    for r in runs:
        method, seed = r["method"], r["seed"]
        key = (method, seed)
        run_dir = Path(f"work/pretraining/runs/{method}_s{seed}")
        events = [json.loads(line) for line in (ROOT / run_dir / "train.jsonl").read_text().splitlines()]
        train = [e for e in events if e["event"] == "train"]
        assert len(train) == 160 and [e["step"] for e in train] == list(range(10, 1601, 10))
        assert events[-1]["event"] == "run_complete" and events[-1]["step"] == 1600
        val = [e for e in events if e["event"] == "validation" and e["step"] == 1600][-1]
        config = read(run_dir / "run.json")
        cost = costs[key]
        assert cost["evaluated_checkpoint_sha256"] == r["checkpoint_sha256"]
        logs[key] = train
        budgets[method] = config["budget"]
        provenance[f"{method}_s{seed}"] = config
        row = {"method": method, "seed": seed, "step": r["step"], "training_tokens": r["training_tokens"],
               "last_10_step_training_loss": train[-1]["loss"], "validation_nll": val["nll_per_token"],
               "validation_ppl": val["perplexity"], "validation_targets": val["tokens"]}
        for name, result in [("test", r["packed_test"]), ("wiki", r["wiki"]), *[(f"long_{length}", values) for length, values in r["long"].items()]]:
            row.update({f"{name}_nll": result["nll_per_token"], f"{name}_ppl": result["perplexity"], f"{name}_targets": result["tokens"]})
        row.update({"training_loop_seconds": cost["seconds"], "training_loop_tokens_per_second": cost["tokens_per_second"],
                    "rank0_logged_peak_allocated_gib": cost["rank0_logged_peak_allocated_gib"],
                    "rank0_numerical_splits_at_last_train_log": cost["rank0_numerical_splits_at_last_train_log"],
                    "total_parameters": config["budget"]["total_parameters"], "feature_parameters": config["budget"]["feature_parameters"],
                    "cache_bytes_at_1024": r["cache"]["bytes_at_1024"], "checkpoint_sha256": r["checkpoint_sha256"], "evaluation_directory": r["evaluation_directory"]})
        metrics.append(row)
        losses.extend({"method": method, "seed": seed, **{k: v for k, v in e.items() if k != "event"}} for e in train)
        recall.extend({"method": method, "seed": seed, **v} for v in r["recall"])
        bands.extend({"method": method, "seed": seed, "position_band": band, **v} for band, v in r["long_position_bands"].items())
        cache = {"method": method, "seed": seed}
        for k, v in r["cache"].items():
            if isinstance(v, dict):
                cache.update({f"{k}_{k2}": v2 for k2, v2 in v.items()})
            else:
                cache[k] = v
        caches.append(cache)
        for name in ["train.jsonl", "run.json", "validation_step001600.json"]:
            target = OUT / "raw" / f"{method}_s{seed}" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / run_dir / name, target)
        for name, expected in r["artifact_sha256"].items():
            source = ROOT / r["evaluation_directory"] / name
            assert digest(source) == expected
            shutil.copy2(source, OUT / "raw" / f"{method}_s{seed}" / name)
    assert len(losses) == 2720 and len(recall) == 153 and len(bands) == 51
    assert all(r["passed"] for r in caches)
    for name, rows in [("metrics.csv", metrics), ("training_loss.csv", losses), ("recall.csv", recall), ("long_position_bands.csv", bands), ("cache.csv", caches)]:
        write_csv(name, rows)
    (OUT / "run_configurations.json").write_text(json.dumps(provenance, indent=2) + "\n")
    package_names = ["torch", "torch_npu", "torchvision", "triton-ascend", "flash-linear-attention", "fla-core", "transformers", "datasets", "numpy", "tokenizers", "pyarrow", "matplotlib", "ad-kernel-ascend"]
    environment = {"python": sys.version, "executable": sys.executable, "packages": {name: importlib.metadata.version(name) for name in package_names}}
    (OUT / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    partial = [json.loads(line) for line in (ROOT / "work/pretraining/runs/elu64_s29/train.jsonl").read_text().splitlines()]
    partial = [e for e in partial if e["event"] == "train"]
    assert partial[-1]["step"] == 100
    write_csv("incomplete_elu64_seed29_loss.csv", [{"method": "elu64", "seed": 29, **e} for e in partial])

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .2, "axes.spines.top": False, "axes.spines.right": False})
    colors = dict(zip(ORDER, plt.get_cmap("tab10").colors))
    for seed in [11, 29]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
        for method in ORDER:
            points = logs.get((method, seed))
            if points is None:
                continue
            for ax in axes:
                ax.plot([p["tokens"] / 1e6 for p in points], [p["loss"] for p in points], label=LABELS[method], color=colors[method], linewidth=1.5)
        axes[0].set(title="Full recorded trajectory", xlim=(0, 106), ylim=(3.8, 10.6))
        axes[1].set(title="Later training (zoom)", xlim=(20, 106), ylim=(3.85, 5.75))
        for ax in axes:
            ax.set_xlabel("Training target tokens (millions)")
            ax.set_ylabel("Next-token cross entropy (nats/token)")
        fig.suptitle(f"Seed {seed} | completed 104.858M-token runs", fontsize=15, y=.99)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=5 if seed == 11 else 4, frameon=False, bbox_to_anchor=(.5, .025))
        fig.text(.5, .004, "Each point is the logged mean over 10 optimizer steps and both NPU ranks; no additional smoothing.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .14, 1, .94))
        save_figure(fig, f"training_loss_seed{seed}")
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.8), sharex=True, sharey=True)
    for method, ax in zip(ORDER, axes.flat):
        for seed, color in [(11, "#2366a9"), (29, "#d86c18")]:
            points = logs.get((method, seed))
            if points:
                ax.plot([p["tokens"] / 1e6 for p in points], [p["loss"] for p in points], label=f"seed {seed}", color=color, linewidth=1.3)
            else:
                ax.text(.98, .94, "seed 29: not completed", transform=ax.transAxes, ha="right", va="top", fontsize=8)
        ax.set(title=LABELS[method], xlim=(20, 106), ylim=(3.85, 5.75))
    for ax in axes[1]:
        ax.set_xlabel("Training tokens (millions)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Training cross entropy")
    fig.suptitle("Per-method seed comparison | later training, 20M-104.858M tokens", fontsize=16)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .05, 1, .94))
    save_figure(fig, "training_loss_by_method")

    generated = {}
    headers = ["方法", "末10步训练损失", "验证 PPL", "主测试 PPL", "Wiki PPL", "长文1k PPL", "长文4k PPL", "长文8k PPL"]
    for seed in [11, 29]:
        generated[f"METRICS_{seed}"] = table(headers, [[LABELS[r["method"]], *[f"{r[k]:.4f}" for k in ["last_10_step_training_loss", "validation_ppl", "test_ppl", "wiki_ppl", "long_1024_ppl", "long_4096_ppl", "long_8192_ppl"]]] for r in metrics if r["seed"] == seed])
        generated[f"RECALL_{seed}"] = table(["方法", "1k/8", "1k/32", "1k/64", "4k/8", "4k/32", "4k/64", "8k/8", "8k/32", "8k/64"], [[LABELS[r["method"]], *[f"{v['candidate_accuracy'] * 100:.2f}" for v in r["recall"]]] for r in runs if r["seed"] == seed])
    generated["PARAMETERS"] = table(["方法", "总可训练参数", "feature可训练参数", "特征宽度m", "B1/1k实际cache MiB"], [[LABELS[m], f"{budgets[m]['total_parameters']:,}", f"{budgets[m]['feature_parameters']:,}", "不适用" if m == "softmax" else budgets[m]["feature_dim"], f"{next(r for r in metrics if r['method'] == m)['cache_bytes_at_1024'] / 2**20:.6f}"] for m in ORDER])
    generated["COST"] = table(["方法", "seed", "训练循环分钟", "循环平均tokens/s", "rank0峰值GiB", "rank0日志数值分段次数"], [[LABELS[r["method"]], r["seed"], f"{r['training_loop_seconds']/60:.3f}", f"{r['training_loop_tokens_per_second']:.1f}", f"{r['rank0_logged_peak_allocated_gib']:.3f}", f"{r['rank0_numerical_splits_at_last_train_log']:,}"] for r in metrics])
    generated["BANDS"] = table(["方法", "seed", "8k执行：目标0–1024 PPL", "目标1024–4096 PPL", "目标4096–8192 PPL"], [[LABELS[r["method"]], r["seed"], *[f"{r['long_position_bands'][b]['perplexity']:.4f}" for b in ["0:1024", "1024:4096", "4096:8192"]]] for r in runs])
    generated["COMPARISONS"] = table(["AD对照方法", "匹配seed", "对照−AD 主测试NLL", "AD相对PPL降低(%)"], [[LABELS[r["other"]], "/".join(map(str, r["seeds"])), f"{r['other_minus_ad_nll']:+.6f}", f"{r['ad_relative_ppl_reduction_percent']:+.4f}"] for r in summary["packed_comparisons"]])
    generated["CACHE"] = table(["方法", "seed", "续接129相对RMS", "续接129输出KL", "解码1025相对RMS", "解码1025输出KL", "通过"], [[LABELS[r["method"]], r["seed"], f"{r['cache']['continued_129']['relative_rms']:.6g}", f"{r['cache']['continued_129']['mean_output_kl']:.6g}", f"{r['cache']['decode_1025']['relative_rms']:.6g}", f"{r['cache']['decode_1025']['mean_output_kl']:.6g}", "是"] for r in runs])
    (OUT / "generated_tables.json").write_text(json.dumps(generated, ensure_ascii=False, indent=2) + "\n")

    source_paths = [Path("work/pretraining/stopped_by_user.json"), Path("work/pretraining/reports/summary.json"), Path("work/pretraining/reports/training_cost.json"), Path("work/pretraining/reports/artifact_audit.json"), Path("work/pretraining/data/fineweb_edu_v1/manifest.json"), Path("work/pretraining/evaluation/long_v1/manifest.json"), Path("work/pretraining/evaluation/transfer_recall_v1/manifest.json"), *[Path(p) for p in next(iter(provenance.values()))["binding"]["code_sha256"]], *[p.relative_to(ROOT) for p in (ROOT / "configs").glob("*.json")]]
    source_paths += [Path("experiments") / name for name in ["prepare_pretraining_data.py", "evaluate_pretrained.py", "inference_runtime.py", "summarize_pretrained.py", "run_pretraining_queue.py", "export_stopped_report.py"]]
    hashes = {}
    for relative in dict.fromkeys(source_paths):
        source, target = ROOT / relative, OUT / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = digest(source)
    (OUT / "source_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps({"report_directory": str(OUT), "completed_runs": len(metrics), "loss_points": len(losses), "recall_conditions": len(recall), "position_bands": len(bands), "cache_checks": len(caches)}))


if __name__ == "__main__":
    main()
