"""Join final-checkpoint quality with verified inference timing and cache costs."""
import hashlib
import fcntl
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "work/pretraining"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_timing(row, batch, decode_tokens, measurements):
    for key, tokens in [("prefill", batch * row["length"]), ("decode", batch * decode_tokens)]:
        timing = row[key]
        samples = timing["samples_seconds"]
        assert len(samples) == measurements and all(math.isfinite(x) and x > 0 for x in samples)
        median = statistics.median(samples)
        assert math.isclose(timing["median_seconds"], median, rel_tol=1e-12)
        assert math.isclose(timing["tokens_per_second"], tokens / median, rel_tol=1e-12)
    assert math.isclose(row["decode_milliseconds_per_step"], row["decode"]["median_seconds"] / decode_tokens * 1000, rel_tol=1e-12)


def main():
    protocol = json.loads((ROOT / "configs/pretraining_protocol_v2.json").read_text())
    training_hash = digest(ROOT / "configs/pretraining_protocol_v2.json")
    benchmark = json.loads((ROOT / "configs/inference_benchmark_v1.json").read_text())
    sources = [ROOT / "experiments/benchmark_inference.py", ROOT / "experiments/inference_runtime.py", ROOT / "configs/inference_benchmark_v1.json"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    code_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:12]
    quality = json.loads((BASE / "reports/summary.json").read_text())
    assert quality["protocol_sha256"] == training_hash
    final_quality = {r["method"]: r for r in quality["runs"] if r["seed"] == 11 and r["step"] == 16384}
    results = []
    for method in protocol["job_order"]:
        for batch in benchmark["batches"]:
            path = BASE / "efficiency" / f"{method}_s11_step016384" / code_id / f"batch{batch}.json"
            if not path.exists():
                continue
            box = json.loads(path.read_text())
            binding = box["binding"]
            assert binding["training"]["protocol_sha256"] == training_hash
            assert binding["training"]["method"] == method and binding["training"]["seed"] == 11
            assert binding["benchmark_code_sha256"] == hashes and binding["batch"] == batch
            assert method in final_quality and binding["checkpoint_sha256"] == final_quality[method]["checkpoint_sha256"]
            assert binding["device"] == ("npu:0" if batch == 1 else "npu:1")
            for source, expected in binding["training"]["code_sha256"].items():
                assert digest(ROOT / source) == expected
            rows = box["results"]
            assert sorted(row["length"] for row in rows) == benchmark["lengths"]
            config, budget = box["budget"]["config"], box["budget"]
            for row in rows:
                assert row["batch"] == batch
                verify_timing(row, batch, benchmark["decode_tokens"], benchmark["measurements"])
                if config["method"] == "softmax":
                    bytes_per_token = batch * config["layers"] * config["width"] * 2 * 2
                    before = bytes_per_token * row["length"]
                    after = bytes_per_token * (row["length"] + benchmark["decode_tokens"])
                else:
                    before = after = budget["linear_state_scalars_per_sequence"] * batch * 4
                assert row["cache_bytes_at_context"] == before and row["cache_bytes_after_decode"] == after
                assert row["prefill_peak_allocated_bytes"] >= before and row["decode_peak_allocated_bytes"] >= after
            results.append({"method": method, "batch": batch, "budget": budget, "results": rows,
                            "test_perplexity": final_quality[method]["packed_test"]["perplexity"],
                            "checkpoint_sha256": binding["checkpoint_sha256"], "artifact": str(path.relative_to(ROOT)),
                            "artifact_sha256": digest(path), "torch": box["torch"], "torch_npu": box["torch_npu"],
                            "ascend_visible_devices": box["ascend_visible_devices"]})
    output = {"training_protocol_sha256": training_hash, "benchmark_code_sha256": hashes, "summary_source_sha256": digest(Path(__file__)),
              "complete_benchmark_jobs": len(results), "expected_benchmark_jobs": 20, "runs": results,
              "scope": benchmark["scope"], "precision": benchmark["precision"]}
    target = BASE / "reports/efficiency.json"
    temporary = target.with_suffix(".json.partial")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)
    lines = ["# 最终模型的推理质量与效率", "", f"已完成 {len(results)}/20 项预定独占设备基准（十种方法，各 B1/B4）。未完成时，本表仅是阶段结果。", "",
             "全部为 seed11、1.074B tokens 固定检查点。骨干为 BF16 autocast、FP32 主权重；线性 feature/状态/attention 为 FP32，softmax SDPA 及 K/V cache 为 BF16。没有权重量化或图捕获。", "",
             "下表展示 8k context。预填充含最后一个位置的词表投影及紧凑缓存构建；解码是固定输入 tokens 的 128 次连续单步计算，包含词表投影，计时不含预填充、采样及 tokenizer。所有数字取 3 次预热后 7 次计时的中位数。", ""]
    for batch in benchmark["batches"]:
        lines += [f"## Batch {batch}", "", "| 方法 | 主测试 PPL | 参数量 M | 预填充 ms | 解码 ms/step | 解码 tokens/s | 缓存 MiB | 解码峰值 GiB |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for result in results:
            if result["batch"] != batch:
                continue
            row = next(r for r in result["results"] if r["length"] == 8192)
            lines.append(f"| {result['method']} | {result['test_perplexity']:.4f} | {result['budget']['total_parameters']/1e6:.3f} | "
                         f"{row['prefill']['median_seconds']*1000:.3f} | {row['decode_milliseconds_per_step']:.3f} | "
                         f"{row['decode']['tokens_per_second']:.1f} | {row['cache_bytes_at_context']/2**20:.3f} | {row['decode_peak_allocated_bytes']/2**30:.3f} |")
        lines.append("")
    lines += ["缓存按实际保留 tensor storage 计数，含线性数值缩放；峰值是 PyTorch allocated memory，包含权重和临时张量，不能与仅缓存大小混用。reserved memory 和各次原始时长另行保存。", "",
              "1k/4k/8k 全部记录、样本时长、数值分段次数、设备快照及数据/checkpoint/source 哈希见 `work/pretraining/reports/efficiency.json` 及其原始产物。测量只代表本次输入、执行实现和两张 910B；最后预算只有一个训练种子，不证明普遍最优。", ""]
    report = ROOT / "reports/EFFICIENCY_RESULTS.zh.md"
    temporary = report.with_suffix(".md.partial")
    temporary.write_text("\n".join(lines))
    temporary.replace(report)
    if results:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"svg.fonttype": "none", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
        colors = dict(zip(protocol["job_order"], plt.get_cmap("tab10").colors))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for result in results:
            if result["batch"] != 1:
                continue
            row = next(r for r in result["results"] if r["length"] == 8192)
            for ax, value in zip(axes, [row["cache_bytes_at_context"] / 2**20, row["decode_milliseconds_per_step"]]):
                ax.scatter(value, result["test_perplexity"], label=result["method"], color=colors[result["method"]], s=55, edgecolors="white", linewidths=.5)
        for ax, label in zip(axes, ["Retained cache at 8k context (MiB)", "Decode latency at 8k context (ms/step)"]):
            ax.set_xlabel(label)
            ax.set_ylabel("Packed test PPL at 1.074B tokens")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .02), ncol=5)
        fig.suptitle("Measured quality/cost tradeoffs — seed 11, batch 1 (lower is better on both axes)")
        fig.tight_layout(rect=(0, .16, 1, .93))
        figure_dir = BASE / "reports/figures"
        figure_dir.mkdir(exist_ok=True)
        for extension in ["png", "svg"]:
            fig.savefig(figure_dir / f"quality_efficiency.{extension}", dpi=180, bbox_inches="tight")
        plt.close(fig)
    print(json.dumps({"event": "efficiency_summary", "complete_benchmark_jobs": len(results)}), flush=True)


if __name__ == "__main__":
    (BASE / "reports").mkdir(exist_ok=True)
    with (BASE / "reports/.efficiency.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
