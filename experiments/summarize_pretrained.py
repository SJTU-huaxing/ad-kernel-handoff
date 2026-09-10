"""Summarize only complete, current-protocol fixed-budget evaluation shards."""
import hashlib
import fcntl
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "work/pretraining"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(records):
    count = sum(row["tokens"] for row in records)
    loss = sum(row["nll_sum"] for row in records)
    return {"tokens": count, "nll_sum": loss, "nll_per_token": loss / count, "perplexity": math.exp(loss / count)}


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    protocol_hash = digest(ROOT / "configs/pretraining_protocol_v2.json")
    protocol = json.loads((ROOT / "configs/pretraining_protocol_v2.json").read_text())
    expected_runs = {(method, seed, 1600) for method in protocol["methods"]
                     for seed in [protocol["primary_seed"], *protocol["additional_seeds"]]}
    expected_runs.update((method, protocol["primary_seed"], 16384) for method in protocol["methods"])
    code_files = [ROOT / "experiments/evaluate_pretrained.py", ROOT / "experiments/inference_runtime.py", ROOT / "configs/pretrained_evaluation_v1.json"]
    code_hashes = {str(p.relative_to(ROOT)): digest(p) for p in code_files}
    code_id = hashlib.sha256(json.dumps(code_hashes, sort_keys=True).encode()).hexdigest()[:12]
    data = json.loads((BASE / "data/fineweb_edu_v1/manifest.json").read_text())
    assert digest(BASE / "data/fineweb_edu_v1/manifest.json") == protocol["data_manifest_sha256"]
    long = json.loads((BASE / "evaluation/long_v1/manifest.json").read_text())
    extra = json.loads((BASE / "evaluation/transfer_recall_v1/manifest.json").read_text())
    case_count = extra["recall_tables"]
    expected = {"packed": set(range((data["tokens"]["test"] - 1) // 1024)),
                "wiki": set(range(extra["wiki_documents"])),
                "long": set(itertools.product(range(len(long["records"])), [1024, 4096, 8192])),
                "recall": set(itertools.product(range(case_count), [1024, 4096, 8192]))}
    complete, pending, details = [], [], {}
    for directory in sorted((BASE / "results").glob(f"*/{code_id}")):
        required = [directory / f"{stage}.rank{rank}of2.json" for stage in ["packed", "wiki", "long", "recall"] for rank in [0, 1]]
        required.append(directory / "cache.rank0of2.json")
        if not all(p.exists() for p in required):
            pending.append({"directory": str(directory.relative_to(ROOT)), "missing": [p.name for p in required if not p.exists()]})
            continue
        boxes = [json.loads(p.read_text()) for p in required]
        binding = boxes[0]["binding"]
        if binding["training"]["protocol_sha256"] != protocol_hash:
            continue
        assert not binding["training"]["synthetic"]
        assert binding["training"]["data_manifest_sha256"] == protocol["data_manifest_sha256"]
        assert binding["tokens"] == binding["step"] * protocol["sequence_length"] * protocol["global_batch_sequences"]
        for source, expected_hash in binding["training"]["code_sha256"].items():
            assert digest(ROOT / source) == expected_hash, source
        for box in boxes:
            candidate = box["binding"]
            assert {k: v for k, v in candidate.items() if k != "rank"} == {k: v for k, v in binding.items() if k != "rank"}
            assert candidate["evaluation_code_sha256"] == code_hashes
        assert binding["long_manifest_sha256"] == digest(BASE / "evaluation/long_v1/manifest.json")
        assert binding["extra_manifest_sha256"] == digest(BASE / "evaluation/transfer_recall_v1/manifest.json")
        records = {}
        for stage in expected:
            rows = [row for box in boxes if box["stage"] == stage for row in box["result"]["records"]]
            indices = [(r["index"], r["length"]) if stage in ["long", "recall"] else r["index"] for r in rows]
            assert set(indices) == expected[stage] and len(indices) == len(set(indices)), (directory, stage)
            records[stage] = sorted(rows, key=lambda row: (row["index"], row.get("length", 0)))
        cache = next(box["result"] for box in boxes if box["stage"] == "cache")
        assert cache["passed"]
        summary = {"method": binding["training"]["method"], "seed": binding["training"]["seed"],
                   "step": binding["step"], "training_tokens": binding["tokens"], "checkpoint_sha256": binding["checkpoint_sha256"],
                   "evaluation_directory": str(directory.relative_to(ROOT)), "packed_test": aggregate(records["packed"]),
                   "wiki": aggregate(records["wiki"]), "long": {str(length): aggregate([r for r in records["long"] if r["length"] == length]) for length in [1024, 4096, 8192]},
                   "cache": cache, "artifact_sha256": {p.name: digest(p) for p in required}}
        summary["long_position_bands"] = {f"{a}:{b}": aggregate([band for row in records["long"] if row["length"] == 8192
                                                               for band in row["position_bands"] if band["start"] == a])
                                          for a, b in [(0, 1024), (1024, 4096), (4096, 8192)]}
        summary["recall"] = []
        for length, pairs in itertools.product([1024, 4096, 8192], [8, 32, 64]):
            subset = [row for row in records["recall"] if row["length"] == length and row["pairs"] == pairs]
            assert len(subset) == 24
            summary["recall"].append({"length": length, "pairs": pairs, "cases": len(subset), "chance": 1 / pairs,
                                      **{field: float(np.mean([row[field] for row in subset])) for field in
                                         ["candidate_accuracy", "gold_log_probability", "gold_candidate_rank", "unrestricted_correct"]}})
        key = (summary["method"], summary["seed"], summary["step"])
        assert key in expected_runs, key
        assert key not in details
        details[key] = records
        complete.append(summary)
    # Conditional paired-document intervals. A partial seed set is reported
    # explicitly; it never stands in for the three-seed milestone comparison.
    comparisons, packed_comparisons = [], []
    for step in [1600, 16384]:
        for other in ["exp64", "hh_exp128", "hh_softmax128", "hh_exp384", "favor64", "favor256", "elu64", "softplus64", "softmax"]:
            seeds = [seed for seed in [11, 29, 47] if ("ad64", seed, step) in details and (other, seed, step) in details]
            if not seeds:
                continue
            packed_deltas = []
            for seed in seeds:
                a = aggregate(details[("ad64", seed, step)]["packed"])
                b = aggregate(details[(other, seed, step)]["packed"])
                assert a["tokens"] == b["tokens"]
                packed_deltas.append(b["nll_per_token"] - a["nll_per_token"])
            packed_delta = float(np.mean(packed_deltas))
            packed_comparisons.append({"ad": "ad64", "other": other, "step": step, "seeds": seeds,
                                       "other_minus_ad_nll": packed_delta,
                                       "per_seed_delta": dict(zip(map(str, seeds), packed_deltas)),
                                       "ad_relative_ppl_reduction_percent": -math.expm1(-packed_delta) * 100,
                                       "scope": "Equal-weight mean of paired-seed test NLL differences. PPL ratio uses geometric means over exactly those matched seeds. No document interval for the packed stream."})
            for stage, length in [("wiki", None), ("long", 1024), ("long", 4096), ("long", 8192)]:
                delta, token_counts, seed_deltas = [], None, []
                for seed in seeds:
                    a = [r for r in details[("ad64", seed, step)][stage] if length is None or r["length"] == length]
                    b = [r for r in details[(other, seed, step)][stage] if length is None or r["length"] == length]
                    assert [r["index"] for r in a] == [r["index"] for r in b]
                    assert [r["tokens"] for r in a] == [r["tokens"] for r in b]
                    token_counts = np.asarray([r["tokens"] for r in a], dtype=np.float64)
                    difference = np.asarray([rb["nll_sum"] - ra["nll_sum"] for ra, rb in zip(a, b)])
                    delta.append(difference)
                    seed_deltas.append(float(difference.sum() / token_counts.sum()))
                mean_delta = np.mean(delta, axis=0)
                rng = np.random.default_rng(131921)
                sample = rng.integers(len(mean_delta), size=(10000, len(mean_delta)))
                boot = mean_delta[sample].sum(1) / token_counts[sample].sum(1)
                comparison = {"ad": "ad64", "other": other, "step": step, "seeds": seeds,
                                    "stage": stage, "length": length, "other_minus_ad_nll": float(mean_delta.sum() / token_counts.sum()),
                                    "per_seed_delta": dict(zip(map(str, seeds), seed_deltas)),
                                    "conditional_document_bootstrap_95ci": np.quantile(boot, [.025, .975]).tolist(),
                                    "scope": "Paired document resampling conditional on the listed trained seeds; positive delta favors AD. Does not quantify general training-seed uncertainty."}
                if stage == "long":
                    # Several long documents share a hostname. Also resample
                    # complete hostname groups so their correlation is retained.
                    hosts = [long["records"][row["index"]]["hostname"] for row in a]
                    groups = sorted(set(hosts))
                    group_delta = np.asarray([sum(mean_delta[i] for i, host in enumerate(hosts) if host == group) for group in groups])
                    group_tokens = np.asarray([sum(token_counts[i] for i, host in enumerate(hosts) if host == group) for group in groups])
                    clusters = np.random.default_rng(251391).integers(len(groups), size=(10000, len(groups)))
                    clustered = group_delta[clusters].sum(1) / group_tokens[clusters].sum(1)
                    comparison["hostname_clusters"] = len(groups)
                    comparison["conditional_hostname_cluster_bootstrap_95ci"] = np.quantile(clustered, [.025, .975]).tolist()
                comparisons.append(comparison)
    result = {"protocol_sha256": protocol_hash, "evaluation_code_sha256": code_hashes, "summary_source_sha256": digest(Path(__file__)), "complete_checkpoints": len(complete),
              "expected_checkpoints": len(expected_runs), "runs": complete, "pending": pending,
              "packed_comparisons": packed_comparisons, "paired_comparisons": comparisons}
    (BASE / "reports").mkdir(exist_ok=True)
    write_json(BASE / "reports/summary.json", result)
    lines = ["# 全层预训练固定预算评测", "", f"当前完成 {len(complete)}/40 个预定检查点的完整测试评测。训练及评测尚未全部完成时，此文件仅是阶段结果。", "",
             "修复版协议：`configs/pretraining_protocol_v2.json`。初版数值故障及其轨迹单独归档，见 `reports/NUMERICAL_STABILITY.zh.md`。", "",
             "1,600 步为 104,857,600 tokens 的早期里程碑；16,384 步为 1,073,741,824 tokens 的完整预算。三个种子均使用完整预算学习率日程，不能将早期里程碑称作已收敛的 100M 训练。", "",
             "| 方法 | seed | 步数 | 主测试 PPL | WikiText PPL | 同篇长文 1k PPL | 4k PPL | 8k PPL |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(complete, key=lambda r: (r["step"], r["method"], r["seed"])):
        metrics = [row["packed_test"]["perplexity"], row["wiki"]["perplexity"], *[row["long"][str(t)]["perplexity"] for t in [1024, 4096, 8192]]]
        lines.append(f"| {row['method']} | {row['seed']} | {row['step']} | " + " | ".join(f"{x:.4f}" for x in metrics) + " |")
    lines += ["", "主测试为 hostname 隔离的 FineWeb-Edu 固定 1,024-token 块。长文是另行冻结的同一批 64 篇连贯文档；不同长度前缀的目标文本也不同，不能把 PPL 差值全部归因于上下文长度。WikiText 是语料风格迁移检查，精确文档去重不能排除引文或近似重叠。", "",
              "关联回忆包含 8/32/64 对随机 key-value 表和 1k/4k/8k 上下文，报告候选准确率、正确 token 概率和随机基准。它同时依赖格式理解与检索能力；接近随机不证明模型结构无法学会关联回忆。", "",
              "AD 与各控制在相同预算、相同种子集合上的比较表见 [AD_COMPARISONS.zh.md](AD_COMPARISONS.zh.md)。逐检查点指标、缓存字节数、关联回忆、文档配对区间和所有输入产物哈希见 `work/pretraining/reports/summary.json`。正的 other−AD NLL 差表示 AD 更好；bootstrap 区间只条件于列出的训练种子。长文另报告按 hostname 整组重采样的区间，以保留同站点文档之间的相关性。正式效率结果须独占设备计时，质量任务耗时不作为模型吞吐。", ""]
    report = ROOT / "reports/PRETRAINING_RESULTS.zh.md"
    temporary = report.with_suffix(".md.partial")
    temporary.write_text("\n".join(lines))
    temporary.replace(report)
    lines = ["# AD 与同预算控制的配对比较", "", f"当前完整质量评测 {len(complete)}/{len(expected_runs)}；下表仅使用双方都完成的相同种子。", "",
             "ΔNLL 定义为控制−AD，正值表示 AD 更好。主测试 PPL 相对降低由相同种子集合上的几何平均 PPL 比率计算，正值为降低、负值为升高。早期三个种子使用完整预算学习率日程，最终预算只有 seed11。", ""]
    for step, label in [(1600, "104,857,600 tokens：早期里程碑"), (16384, "1,073,741,824 tokens：最终预算")]:
        lines += [f"## {label}", ""]
        selected = [row for row in packed_comparisons if row["step"] == step]
        if not selected:
            lines += ["尚无双方均完成的配对结果。", ""]
            continue
        lines += ["| 控制 | 配对 seeds | 主测试 ΔNLL | AD PPL 相对降低 | 各 seed ΔNLL |",
                  "|---|---|---:|---:|---|"]
        for row in selected:
            seeds = ", ".join(map(str, row["seeds"]))
            individual = "; ".join(f"{seed}: {delta:+.6f}" for seed, delta in row["per_seed_delta"].items())
            lines.append(f"| {row['other']} | {seeds} | {row['other_minus_ad_nll']:+.6f} | {row['ad_relative_ppl_reduction_percent']:+.3f}% | {individual} |")
        lines += ["", "| 控制 | 配对 seeds | 评测 | ΔNLL | 文档 bootstrap 95% 区间 | hostname 整组 bootstrap 95% 区间 |",
                  "|---|---|---|---:|---|---|"]
        for row in comparisons:
            if row["step"] != step:
                continue
            name = "WikiText" if row["stage"] == "wiki" else f"长文 {row['length'] // 1024}k"
            seeds = ", ".join(map(str, row["seeds"]))
            lo, hi = row["conditional_document_bootstrap_95ci"]
            group = row.get("conditional_hostname_cluster_bootstrap_95ci")
            group_interval = f"[{group[0]:+.6f}, {group[1]:+.6f}]" if group else "—"
            lines.append(f"| {row['other']} | {seeds} | {name} | {row['other_minus_ad_nll']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {group_interval} |")
        lines.append("")
    lines += ["每个区间用 10,000 次配对重采样；先对当前共同种子的文档 NLL 差取平均，再重采样文档或完整 hostname 组。区间仅条件于所列训练种子，不量化一般训练随机性，也未做多重比较校正。主测试为打包 token 流，不在此冒用文档独立性给出区间。", "",
              "训练、数据、代码及检查点绑定和逐种子结果见 `work/pretraining/reports/summary.json`。评测不参与模型、学习率或训练预算选择；最终结论需待预定对比完成。", ""]
    report = ROOT / "reports/AD_COMPARISONS.zh.md"
    temporary = report.with_suffix(".md.partial")
    temporary.write_text("\n".join(lines))
    temporary.replace(report)
    print(json.dumps({"event": "summary", "complete_checkpoints": len(complete), "paired_comparisons": len(comparisons)}), flush=True)


if __name__ == "__main__":
    (BASE / "reports").mkdir(exist_ok=True)
    with (BASE / "reports/.summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
