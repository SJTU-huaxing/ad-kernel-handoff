"""Report matched-target context effects with paired uncertainty estimates."""
import hashlib
import fcntl
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "work/pretraining"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_intervals(differences, hostnames):
    differences = np.asarray(differences, dtype=np.float64)
    n = len(differences)
    assert n == len(hostnames) and n > 1
    draws = np.random.default_rng(88193).integers(n, size=(10000, n))
    document = differences[draws].mean(1)
    groups = sorted(set(hostnames))
    sums = np.asarray([sum(d for d, h in zip(differences, hostnames) if h == group) for group in groups])
    counts = np.asarray([hostnames.count(group) for group in groups])
    cluster_draws = np.random.default_rng(91031).integers(len(groups), size=(10000, len(groups)))
    cluster = sums[cluster_draws].sum(1) / counts[cluster_draws].sum(1)
    return {"nll_difference": float(differences.mean()), "perplexity_ratio": math.exp(float(differences.mean())),
            "document_bootstrap_95ci": np.quantile(document, [.025, .975]).tolist(),
            "hostname_cluster_bootstrap_95ci": np.quantile(cluster, [.025, .975]).tolist(),
            "documents": n, "hostname_clusters": len(groups)}


def main():
    protocol = ROOT / "configs/context_control_v1.json"
    config = json.loads(protocol.read_text())
    files = [ROOT / "experiments/evaluate_context_controls.py", protocol]
    code_hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
    code_id = hashlib.sha256(json.dumps(code_hashes, sort_keys=True).encode()).hexdigest()[:12]
    training_hash = digest(ROOT / "configs/pretraining_protocol_v2.json")
    training_config = json.loads((ROOT / "configs/pretraining_protocol_v2.json").read_text())
    long_manifest = BASE / "evaluation/long_v1/manifest.json"
    documents = json.loads(long_manifest.read_text())["records"]
    hostnames = [row["hostname"] for row in documents]
    conditions = [row["name"] for row in config["conditions"]]
    reference_targets, runs = {}, []
    contrasts = [
        ("context8192_full", "context1024_reset"),
        ("context4096_reset", "context1024_reset"),
        ("context1024_document_positions", "context1024_reset"),
        ("context4096_document_positions", "context4096_reset"),
        ("context8192_full", "context1024_document_positions"),
        ("context8192_full", "context4096_document_positions")]
    for method in training_config["job_order"]:
        directory = BASE / "context_controls" / f"{method}_s11_step016384" / code_id
        paths = [directory / f"rank{rank}.json" for rank in [0, 1]]
        if not all(path.exists() for path in paths):
            continue
        boxes = [json.loads(path.read_text()) for path in paths]
        binding = boxes[0]["binding"]
        assert binding["training"]["method"] == method and binding["training"]["protocol_sha256"] == training_hash
        for box in boxes:
            other = box["binding"]
            assert {k: v for k, v in other.items() if k != "rank"} == {k: v for k, v in binding.items() if k != "rank"}
            assert other["source_sha256"] == code_hashes and other["long_manifest_sha256"] == digest(long_manifest)
        records = [row for box in boxes for row in box["records"]]
        assert len(records) == len(documents) * len(conditions)
        lookup = {condition: {} for condition in conditions}
        for row in records:
            assert row["tokens"] == 1024 and row["index"] not in lookup[row["condition"]]
            index = row["index"]
            assert reference_targets.setdefault(index, row["target_sha256"]) == row["target_sha256"]
            lookup[row["condition"]][index] = row
        for condition in conditions:
            assert set(lookup[condition]) == set(range(len(documents)))
        nll = {condition: np.asarray([lookup[condition][i]["nll_sum"] / 1024 for i in range(len(documents))]) for condition in conditions}
        result = {"method": method, "seed": 11, "step": 16384, "checkpoint_sha256": binding["checkpoint_sha256"],
                  "source_artifact_sha256": {path.name: digest(path) for path in paths},
                  "conditions": {name: {"nll_per_token": float(values.mean()), "perplexity": math.exp(float(values.mean())), "tokens": len(documents) * 1024}
                                 for name, values in nll.items()},
                  "contrasts": [{"a": a, "b": b, "direction": "Positive A-minus-B NLL means A is worse", **paired_intervals(nll[a] - nll[b], hostnames)} for a, b in contrasts]}
        runs.append(result)
    output = {"protocol_sha256": digest(protocol), "evaluation_source_sha256": code_hashes, "summary_source_sha256": digest(Path(__file__)),
              "training_protocol_sha256": training_hash, "complete_methods": len(runs), "expected_methods": 10, "runs": runs,
              "scope": config["status"], "limitations": config["limitations"]}
    target = BASE / "reports/context_controls.json"
    temporary = target.with_suffix(".json.partial")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)
    lines = ["# 固定目标 token 的上下文与位置诊断", "", f"已完成 {len(runs)}/10 个最终预算模型（seed11、1.074B tokens）。未完成前只作阶段报告。", "",
             "此补充协议在看到 AD 的早期前缀测试后制定，属于探索性诊断。它不参与模型、训练预算或超参数选择。", "",
             "全部条件都预测同一批 64 篇长文中，固定 8k 评测前缀的最后 1,024 个目标 tokens（原文 token 索引 7169..8192）。这不是整篇原文的结尾。reset 表示输入片段从 RoPE 位置 0 开始；document positions 表示保留其在原文中的位置。每项均从空状态开始，模型权重不变。", "",
             "| 方法 | 1k reset PPL | 4k reset PPL | 8k full PPL | 1k document positions PPL | 4k document positions PPL |", "|---|---:|---:|---:|---:|---:|"]
    for run in runs:
        lines.append(f"| {run['method']} | " + " | ".join(f"{run['conditions'][name]['perplexity']:.4f}" for name in conditions) + " |")
    lines += ["", "同 context 长度下 reset 与 document positions 的差异用于诊断绝对位置敏感性；保留 document positions 后与 8k full 比较，用于观察移除较早上下文的影响。相比前缀表，这里不存在目标 token 不同造成的文本难度差异。", "",
              "逐文档配对差异、文档及 hostname 整组 bootstrap 区间保存在 `work/pretraining/reports/context_controls.json`。区间仅条件于这一个训练种子，不能用于宣称对所有训练随机性或模型规模普遍成立。", ""]
    report = ROOT / "reports/CONTEXT_CONTROLS.zh.md"
    temporary = report.with_suffix(".md.partial")
    temporary.write_text("\n".join(lines))
    temporary.replace(report)
    print(json.dumps({"event": "context_summary", "complete_methods": len(runs)}), flush=True)


if __name__ == "__main__":
    (BASE / "reports").mkdir(exist_ok=True)
    with (BASE / "reports/.context_controls.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
