"""Export data-backed training/quality figures; partial results are labelled."""
import json
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "work/pretraining"
OUT = BASE / "reports/figures"
OUT.mkdir(parents=True, exist_ok=True)
protocol = json.loads((ROOT / "configs/pretraining_protocol_v2.json").read_text())
order = protocol["job_order"]
colors = dict(zip(order, plt.get_cmap("tab10").colors))
plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})


def records(path):
    values = []
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return values


def save(figure, name):
    for extension in ["png", "svg"]:
        figure.savefig(OUT / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(figure)


fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
plotted, validation_present = [], False
for method in order:
    path = BASE / "runs" / f"{method}_s11" / "train.jsonl"
    if not path.exists():
        continue
    events = records(path)
    latest_start = max(i for i, row in enumerate(events) if row["event"] in ["start", "resume"])
    current = [row for row in events[latest_start:] if row["event"] == "train"]
    cutoff = current[-1]["step"] if current else events[latest_start]["step"]
    by_step = {row["step"]: row for row in events if row["event"] == "train" and row["step"] <= cutoff}
    train = [by_step[step] for step in sorted(by_step)]
    if not train:
        continue
    plotted.append(method)
    axes[0].plot([r["tokens"] / 1e6 for r in train], [r["loss"] for r in train], color=colors[method], label=method, linewidth=1.2)
    checkpoints = {row["step"] for row in events if row["event"] == "checkpoint"}
    throughput, since_start, previous = {}, 0, 0
    for row in events:
        if row["event"] in ["start", "resume"]:
            since_start, previous = 0, row["step"]
        elif row["event"] == "train":
            since_start += 1
            if since_start > 3 and row["step"] <= cutoff and not any(previous < step <= row["step"] for step in checkpoints):
                throughput[row["step"]] = row
            previous = row["step"]
    measured = [throughput[step] for step in sorted(throughput)]
    axes[1].plot([r["tokens"] / 1e6 for r in measured], [r["interval_tokens_per_second"] / 1000 for r in measured], color=colors[method], linewidth=1.2)
    validation = {row["step"]: row for row in events if row["event"] == "validation" and row["step"] <= cutoff}
    if validation:
        validation_present = True
        rows = [validation[step] for step in sorted(validation)]
        axes[2].plot([r["step"] * 65536 / 1e6 for r in rows], [r["perplexity"] for r in rows], color=colors[method], marker="o", markersize=3, linewidth=1.2)
axes[0].set_ylabel("Training NLL (10-step mean)")
axes[1].set_ylabel("Two-NPU training throughput (k tokens/s)")
axes[2].set_ylabel("Held-out hostname validation PPL")
for ax in axes:
    ax.set_xlabel("Training tokens (millions)")
    ax.grid(alpha=.2)
axes[2].set_yscale("log")
if not validation_present:
    axes[2].set_axis_off()
    axes[2].text(.5, .5, "Validation milestone pending", ha="center", va="center", transform=axes[2].transAxes)
if plotted:
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(plotted)), bbox_to_anchor=(.5, .045))
fig.suptitle("Protocol v2, seed 11 — training progress snapshot (full budget: 1.074B tokens)")
fig.text(.5, .012, "Throughput excludes the first three intervals after each start/resume and intervals crossing checkpoint steps; includes data, HCCL and AdamW.", ha="center", fontsize=8)
fig.tight_layout(rect=(0, .15, 1, .94))
save(fig, "training_progress")

summary_path = BASE / "reports/summary.json"
summary_raw = summary_path.read_bytes() if summary_path.exists() else b'{"runs": []}'
summary = json.loads(summary_raw)
if summary["runs"]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, step, name in zip(axes, [1600, 16384], ["104.858M tokens: early milestone", "1.074B tokens: fixed final budget"]):
        labels, position = [], 0
        for method in order:
            selected = [row for row in summary["runs"] if row["method"] == method and row["step"] == step]
            if not selected:
                continue
            nlls = np.asarray([row["packed_test"]["nll_per_token"] for row in selected])
            mean = nlls.mean()
            std = nlls.std(ddof=1) if len(nlls) > 1 else 0.
            center, low, high = np.exp([mean, mean - std, mean + std])
            ax.errorbar(position, center, yerr=np.asarray([[center-low], [high-center]]), color=colors[method], fmt="o", capsize=4)
            offsets = np.linspace(-.10, .10, len(nlls)) if len(nlls) > 1 else np.zeros(1)
            ax.scatter(position + offsets, np.exp(nlls), color=colors[method], s=12, alpha=.6)
            labels.append(f"{method}\n(n={len(nlls)})")
            position += 1
        ax.set_xticks(range(position), labels, rotation=35, ha="right")
        ax.set_title(name)
        ax.set_ylabel("Packed test PPL")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=.2)
        if not labels:
            ax.set_axis_off()
            ax.text(.5, .5, "No completed test evaluation yet", ha="center", transform=ax.transAxes)
    fig.suptitle(f"Complete quality evaluations: {summary['complete_checkpoints']}/40 — partial until all runs finish")
    fig.text(.5, .045, "Centers: geometric mean PPL over available seeds. Dots: individual seeds. Bars: ±1 sample standard deviation in NLL, not confidence intervals.", ha="center", fontsize=8)
    fig.text(.5, .012, "All seeds use the full-budget LR schedule; early milestones are not separately converged 100M-token runs.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .11, 1, .94))
    save(fig, "test_quality_by_budget")

# Every method is present for this explicitly single-seed early comparison.
# Mixed available-seed counts in the progress figure are not reused here.
early = {row["method"]: row for row in summary["runs"] if row["seed"] == 11 and row["step"] == 1600}
cost_path = BASE / "reports/training_cost.json"
if set(early) == set(order) and cost_path.exists():
    cost_raw = cost_path.read_bytes()
    costs = json.loads(cost_raw)
    assert costs["training_protocol_sha256"] == summary["protocol_sha256"]
    rows = []
    for method in order:
        measured = [r for r in costs["invocations"] if r["method"] == method and r["seed"] == 11
                    and r["status"] == "complete" and r["start_step"] == 0 and r["last_logged_step"] == 1600]
        assert len(measured) == 1, f"A complete uninterrupted early invocation is required for {method}"
        timing, quality = measured[0], early[method]
        assert timing["evaluated_checkpoint_sha256"] == quality["checkpoint_sha256"]
        metadata_raw = (BASE / "runs" / f"{method}_s11/run.json").read_bytes()
        metadata = json.loads(metadata_raw)
        assert metadata["binding"]["protocol_sha256"] == summary["protocol_sha256"]
        rows.append({"method": method, "seed": 11, "training_tokens": quality["training_tokens"],
                     "parameters": metadata["budget"]["total_parameters"],
                     "packed_test_perplexity": quality["packed_test"]["perplexity"], "wiki_perplexity": quality["wiki"]["perplexity"],
                     "long_8192_perplexity": quality["long"]["8192"]["perplexity"],
                     "cache_bytes_at_1024_batch1": quality["cache"]["bytes_at_1024"],
                     "training_loop_seconds": timing["seconds"], "checkpoint_sha256": quality["checkpoint_sha256"],
                     "run_metadata_snapshot_sha256": hashlib.sha256(metadata_raw).hexdigest()})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), sharey=True)
    for row in rows:
        for ax, value in zip(axes, [row["cache_bytes_at_1024_batch1"] / 2**20, row["training_loop_seconds"] / 60]):
            ax.scatter(value, row["packed_test_perplexity"], color=colors[row["method"]], label=row["method"],
                       s=48, edgecolors="white", linewidths=.5, alpha=.9)
    axes[0].set_xscale("log")
    cache_sizes = sorted({r["cache_bytes_at_1024_batch1"] / 2**20 for r in rows})
    axes[0].set_xticks(cache_sizes, [f"{size:.2f}" for size in cache_sizes])
    axes[0].minorticks_off()
    axes[0].set_xlabel("Retained 1k cache (MiB), batch 1, all 12 layers")
    axes[1].set_xlabel("Training loop minutes for 104.858M tokens, two NPUs")
    axes[0].set_ylabel("Packed test PPL")
    for ax in axes:
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .085), ncol=5)
    fig.suptitle("All ten methods at the same early budget — seed 11 only, protocol v2")
    fig.text(.5, .048, "Lower is better on both axes. Training time includes in-loop validation and saving; it is not an inference benchmark.", ha="center", fontsize=8)
    fig.text(.5, .015, "Measured cache: FP32 linear states versus BF16 softmax K/V. Single-seed point estimates; full-budget results remain separate.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .24, 1, .94))
    save(fig, "early_quality_cost_seed11")
    payload = {"scope": "All ten methods, seed11, 104857600 tokens. Same evaluated checkpoint joins quality, actual 1024-token batch1 cache and completed training invocation. Single-seed early results, not final-budget or inference timing.",
               "protocol_sha256": summary["protocol_sha256"], "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "quality_summary_snapshot_sha256": hashlib.sha256(summary_raw).hexdigest(),
               "training_cost_snapshot_sha256": hashlib.sha256(cost_raw).hexdigest(), "rows": rows}
    (OUT / "early_quality_cost_seed11.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = ["# 十种方法的同预算早期质量与成本", "", "全部为 seed11、104,857,600 tokens，避免将不同数量的可用种子混合比较。只有一个种子的早期点估计，不能代替三种子结果或最终 1.074B 预算结果。", "",
             "| 方法 | 参数 M | 主测试 PPL | WikiText PPL | 8k 长文 PPL | 实测 1k 缓存 MiB | 训练循环分钟 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['parameters']/1e6:.3f} | {row['packed_test_perplexity']:.4f} | {row['wiki_perplexity']:.4f} | "
                     f"{row['long_8192_perplexity']:.4f} | {row['cache_bytes_at_1024_batch1']/2**20:.3f} | {row['training_loop_seconds']/60:.3f} |")
    lines += ["", "缓存为 B1、1024 tokens、全部 12 层实际保留 storage，含线性数值缩放状态；softmax K/V 为 BF16，线性状态为 FP32。模型骨干均为 BF16 autocast、FP32 主参数；共享骨干相同，各方法 feature 参数和状态预算另行列出。", "",
              "时间对应从 0 到 1600 的真实训练调用，含中途验证和保存，不含启动前准备和独立质量评测。它不是最终推理计时。更细的时间范围见 [TRAINING_COST.zh.md](TRAINING_COST.zh.md)；相同种子集合的配对统计见 [AD_COMPARISONS.zh.md](AD_COMPARISONS.zh.md)。", "",
              "PNG/SVG 图及逐点来源在 `work/pretraining/reports/figures/early_quality_cost_seed11.*`。每个点按 checkpoint SHA256 联结质量和训练成本。", ""]
    (ROOT / "reports/EARLY_QUALITY_COST.zh.md").write_text("\n".join(lines))
print(json.dumps({"event": "figures", "directory": str(OUT), "training_methods": plotted,
                  "complete_quality_checkpoints": summary.get("complete_checkpoints", 0)}), flush=True)
