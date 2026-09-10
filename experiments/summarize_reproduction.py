"""Summarize only a complete, hash-bound fresh confirmation run."""
import json
from pathlib import Path
import numpy as np
from reproduce_features import ROOT, RUN, digest, save_json

plan = json.loads((RUN / "frozen_evaluation.json").read_text())
source_hash = digest(ROOT / "experiments/assess_reproduction.py")
directory = RUN / "confirmation" / source_hash[:12]
groups = {
    "standard_ad": [name for name in plan["models"] if name.startswith("standard_ad_")],
    "standard_exp": [name for name in plan["models"] if name.startswith("standard_exp_")],
    "matched_zero_ad": [name for name in plan["models"] if name.startswith("matched_zero_ad_")],
    "matched_zero_exp": [name for name in plan["models"] if name.startswith("matched_zero_exp_")],
    "hh_exp": [name for name in plan["models"] if name.startswith("hh_exp_")],
    "hh_softmax": [name for name in plan["models"] if name.startswith("hh_softmax_")],
    "favor": [name for name in plan["models"] if name.startswith("favor_")],
}
records, sources = {}, {}
for split in plan["splits"]:
    for name in plan["models"]:
        path = directory / f"{name}_{split}.json"
        value = json.loads(path.read_text())
        assert value["binding"]["plan_sha256"] == digest(RUN / "frozen_evaluation.json")
        assert value["binding"]["assessment_source_sha256"] == source_hash
        records[name, split] = value
        sources[str(path.relative_to(ROOT))] = digest(path)
summary, paired = {}, {}
for split in plan["splits"]:
    teacher0, teacher1 = [json.loads((directory / f"teacher_{split}_rank{rank}.json").read_text()) for rank in [0, 1]]
    rank_difference = abs(teacher0["ppl"]["nll_per_token"] - teacher1["ppl"]["nll_per_token"])
    assert rank_difference < 1e-6, rank_difference
    summary[split] = {"teacher": {"ppl": teacher0["ppl"]["perplexity"], "rank_nll_difference": rank_difference}}
    for group, names in groups.items():
        assert len(names) == 3
        rows = [records[name, split] for name in names]
        ppl = [r["ppl"]["perplexity"] for r in rows]
        summary[split][group] = {"ppl_mean": float(np.mean(ppl)), "ppl_seed_sd": float(np.std(ppl, ddof=1)),
                                 "ppl_by_seed": ppl, "kl": float(np.mean([r["static"]["summary"]["kl"] for r in rows])),
                                 "tv": float(np.mean([r["static"]["summary"]["tv"] for r in rows])),
                                 "output_nmse": float(np.mean([r["static"]["summary"]["output_nmse"] for r in rows]))}
    for initialization in ["standard", "matched_zero"]:
        differences = []
        for ad, exp in zip(groups[initialization + "_ad"], groups[initialization + "_exp"]):
            a, b = records[ad, split]["ppl"]["documents"], records[exp, split]["ppl"]["documents"]
            assert [r["text_sha256"] for r in a] == [r["text_sha256"] for r in b]
            differences.append([y["nll_sum"] / y["tokens"] - x["nll_sum"] / x["tokens"] for x, y in zip(a, b)])
        differences = np.asarray(differences)
        per_document = differences.mean(0)
        rng = np.random.default_rng(2940)
        boot = per_document[rng.integers(len(per_document), size=(10000, len(per_document)))].mean(1)
        paired[split + "/" + initialization] = {
            "exp_minus_ad_nll": float(per_document.mean()), "by_seed": differences.mean(1).tolist(),
            "conditional_document_bootstrap95": np.quantile(boot, [.025, .975]).tolist(),
            "positive_documents": int((per_document > 0).sum()), "documents": len(per_document),
            "scope": "Paired document resampling conditional on these three trained seeds and this fixed public manifest; positive favors AD"}
result = {"scope": plan["scope"], "summary": summary, "paired": paired, "sources": sources,
          "plan_sha256": digest(RUN / "frozen_evaluation.json")}
save_json(RUN / "summary.json", result)
lines = ["# Ascend 新公开数据复现", "", "这是重建公开数据上的复现。原 GPU token/QKV 清单未提供，不能将绝对指标当作原表格的逐样本复现。",
         "", "固定 Qwen2.5-1.5B，只替换第 14、27 层全部 24 个 Q heads；feature 拟合每次单遍 4,194,304 tokens。模型主体未训练。",
         "", "所有学习率仅由 seed 11 的前 32 验证文档选择；21 个模型和参数哈希均在读取确认集前冻结。", "",
         "| 方法 | 1k KL | 1k PPL（3 seeds 均值） | 8k KL | 8k PPL（3 seeds 均值） |", "|---|---:|---:|---:|---:|"]
for group in groups:
    a, b = summary["confirm_wiki"][group], summary["confirm_long"][group]
    lines.append(f"| {group} | {a['kl']:.6f} | {a['ppl_mean']:.6f} | {b['kl']:.6f} | {b['ppl_mean']:.6f} |")
lines += ["", f"原 softmax 教师 PPL：1k {summary['confirm_wiki']['teacher']['ppl']:.6f}；8k {summary['confirm_long']['teacher']['ppl']:.6f}。",
          "", "长上下文方向在重建数据上重现：两个初始化下的 3 个种子均为 AD 的 8k NLL 更低；标准初始化 23/24 文档、匹配零初始化 20/24 文档的种子平均差有利 AD。1k PPL 的配对文档区间均包含 0，不能声称短文优势。",
          "", "AD 的 8k KL 仍高于 Hedgehog，而 PPL 更低；所有替换方案的 PPL 仍高于原 softmax 教师。静态 attention KL 的排序不能直接代替完整模型质量排序。",
          "", "每个种子的 PPL 由全部下一 token 的总 NLL/有效 token 数求指数，再取种子算术均值。完整逐文档值、种子波动和配对差见 `work/reproduction/public_wikitext_v1/summary.json`。",
          "", "AD/EXP：73536 学习参数/head、m64；Hedgehog 适配：73728 参数/head、m576；FAVOR：0 学习参数、m64。不是所有方法的参数量、状态和计算预算同时相等。",
          "", "静态指标为 CPU FP64。PPL 使用 BF16 Qwen 主体与经真实激活检查的 FP32 归一化分块替换；质量运行的耗时包含 CPU 指标和数据传输，不能当作效率基准。",
          "", "本结果只涉及局部替换，不能外推全层预训练质量；尚不支持普遍最优、理论达界或完整 Hedgehog/Performer 论文复现等声明。"]
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports/ASCEND_REPRODUCTION.zh.md").write_text("\n".join(lines) + "\n")
print(json.dumps({"complete": True, "summary": summary, "paired": paired}), flush=True)
