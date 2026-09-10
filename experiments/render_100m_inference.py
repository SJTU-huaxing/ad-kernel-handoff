"""Render completed inference measurements, preserving all raw samples."""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    base = args.directory.resolve()
    bench = json.loads((base/"benchmark.json").read_text())
    features = json.loads((base/"features.json").read_text())
    assert bench.get("complete") and features.get("complete")
    steps = bench["arguments"]["decode_tokens"]
    def aggregate(rows, **filters):
        picked = [r for r in rows if all(r.get(k) == v for k, v in filters.items())]
        assert picked
        factor = steps if filters.get("stage") == "decode" else 1
        wall = [s["wall_ms"]/factor for r in picked for s in r["samples"]]
        event = [s["event_ms"]/factor for r in picked for s in r["samples"]]
        return {"median_ms":statistics.median(wall), "p25_ms":float(np.quantile(wall,.25)),
                "p75_ms":float(np.quantile(wall,.75)), "median_event_ms":statistics.median(event),
                "round_medians_ms":[r["median_wall_ms"]/factor for r in picked],
                "samples_ms":wall}
    summary = {"model":[], "features":[]}
    for stage, executions, lengths in [
            ("prefill", ["eager"], bench["arguments"]["prefill_lengths"]),
            ("decode", ["eager", "graph"], bench["arguments"]["decode_contexts"])]:
        for batch in bench["arguments"]["batches"]:
            for length in lengths:
                for execution in executions:
                    row = {"stage":stage, "batch":batch, "length":length, "execution":execution}
                    for method in ["ad64", "hedgehog"]:
                        row[method] = aggregate(bench["rows"], method=method, **{k:v for k,v in row.items() if k in ["stage","batch","length","execution"]})
                    row["hedgehog_over_ad_time"] = row["hedgehog"]["median_ms"]/row["ad64"]["median_ms"]
                    summary["model"].append(row)
    for batch in [1,8]:
        for length in [1,2048]:
            for execution in ["eager","graph"]:
                row = {"batch":batch, "length":length, "execution":execution}
                for method in ["ad64","hedgehog"]:
                    row[method] = aggregate(features["rows"], method=method, batch=batch, length=length, execution=execution)
                row["hedgehog_over_ad_time"] = row["hedgehog"]["median_ms"]/row["ad64"]["median_ms"]
                row["hedgehog_over_ad_event_time"] = row["hedgehog"]["median_event_ms"]/row["ad64"]["median_event_ms"]
                summary["features"].append(row)
    (base/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")

    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig, axes = plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for i,batch in enumerate([1,8]):
        for j,(stage,execution) in enumerate([("prefill","eager"),("decode","graph")]):
            ax = axes[j,i]
            rows = [r for r in summary["model"] if r["batch"] == batch and r["stage"] == stage and r["execution"] == execution]
            x = np.arange(len(rows))
            for k,(method,label,color) in enumerate([("ad64","AD (m=64)","#2563eb"),("hedgehog","FLA Hedgehog (m=128)","#ea580c")]):
                ys = [r[method]["median_ms"] for r in rows]
                yerr = [[max(0,r[method]["median_ms"]-r[method]["p25_ms"]) for r in rows],
                        [max(0,r[method]["p75_ms"]-r[method]["median_ms"]) for r in rows]]
                bars=ax.bar(x+(k-.5)*.34,ys,.34,label=label,color=color,yerr=yerr,capsize=3)
                ax.bar_label(bars,fmt="%.2f",padding=3,fontsize=8)
            ax.set_xticks(x,[str(r["length"]) for r in rows])
            ax.set_xlabel("Prompt / cached context length (tokens)")
            ax.set_ylabel("Prompt latency (ms)" if stage == "prefill" else "Decode latency (ms / step)")
            ax.set_title(f"{'Prefill' if stage == 'prefill' else 'NPU graph decode'} | batch {batch}")
            ax.set_ylim(0,ax.get_ylim()[1]*1.14)
            if i == 0 and j == 0: ax.legend(loc="upper left")
    fig.suptitle("100M AD vs official FLA Hedgehog | Ascend 910B3, BF16 backbone / FP32 state\nFresh seed 11; one NPU; median and IQR; lower is faster",fontsize=13)
    fig.savefig(base/"latency.png",dpi=180)
    fig.savefig(base/"latency.pdf")
    plt.close(fig)

    lines = ["# AD 与官方 FLA Hedgehog：约 100M 模型推理时间实测", "",
             f"完成时间：{bench['completed_utc']}。正式训练未启动。", "",
             "本次测量当前已准备的约 100M 架构：AD 101,964,800 参数，Hedgehog 99,127,040 参数；两者共同主干 99,027,200 参数且权重 SHA256 完全相同。均为 seed 11 新初始化权重，没有采用 2B tokens 训练后的模型，因此这里只比较计算性能，不能据此评价生成质量或训练后极端激活的速度。", "",
             "AD 使用每头独立的 Q/K 两层 SiLU MLP，特征维度 64；Hedgehog 使用官方 FLA `HedgehogFeatureMap` 的逐层 Q/K 映射，跨头共享，特征维度 128。Hedgehog 包含官方的 2 倍线性变换、正负拼接及联合 softmax；运行时用等价的 log-softmax 保留数值稳定性。", "",
             "硬件为同一张 Ascend 910B3 64 GB（逻辑 NPU 0，ASCEND_VISIBLE_DEVICES="+str(bench['ascend_visible_devices'])+"）。单个模型依次独占该卡，另一张卡未用于并发计时。两轮顺序 AD→Hedgehog、Hedgehog→AD。BF16 主干计算、FP32 模型主权重及特征/状态；相同 FLA 融合 RMSNorm、残差归一化和 SwiGLU。", "",
             "计时包括完整 12 层、最终归一化、最后位置的 50,257 词表投影，以及可继续使用的紧凑缓存。Prefill 只输出最后位置 logits，符合生成推理需求；不计算每个历史位置的词表 logits。Decode 使用完全相同的真实 FineWeb-Edu 留出集后续 token 驱动，逐步更新全部层状态和 RoPE 位置；属于固定输入的自回归计算基准，未加入采样、文本分词、传输或服务排队。", "",
             f"每个条件每轮 {bench['arguments']['samples']} 次测量，共 {bench['arguments']['samples']*bench['arguments']['rounds']} 次；decode 每次连续 {steps} 步。完整模型表使用同步墙钟时间的中位数，decode 除以步数。预热、编译、图捕获和每轮缓存重置不计时；图执行的输入设备内复制计时。每个样本另存 NPU event 时间、缓存字节数、峰值已分配内存及原始时长。中位数的四分位区间和各轮分别的中位数见 summary.json。", "",
             "`HH/AD` 定义为 Hedgehog 耗时 ÷ AD 耗时：大于 1 表示 AD 更快，小于 1 表示 Hedgehog 更快。", "",
             "## 整段输入（prefill）", "", "| Batch | 输入 tokens | AD ms | Hedgehog ms | HH/AD |", "|---:|---:|---:|---:|---:|"]
    for r in summary["model"]:
        if r["stage"] == "prefill": lines.append(f"| {r['batch']} | {r['length']} | {r['ad64']['median_ms']:.3f} | {r['hedgehog']['median_ms']:.3f} | {r['hedgehog_over_ad_time']:.3f}× |")
    lines += ["", "Prefill 使用已验证的 CANN 分块精确线性注意力路径。当前路径包含数值安全判定及调度开销，本表衡量现有实现，而非两类数学算法的硬件速度上限。", "", "## 带缓存逐 token 解码", "", "| Batch | 缓存 tokens | 执行 | AD ms/步 | Hedgehog ms/步 | HH/AD |", "|---:|---:|---|---:|---:|---:|"]
    for r in summary["model"]:
        if r["stage"] == "decode": lines.append(f"| {r['batch']} | {r['length']} | {r['execution']} | {r['ad64']['median_ms']:.3f} | {r['hedgehog']['median_ms']:.3f} | {r['hedgehog_over_ad_time']:.3f}× |")
    lines += ["", "Batch=1 时，ms/步就是单请求 ms/token；Batch=8 时，每一步为 8 个请求各生成一个 token，总吞吐为 8,000 ÷ ms/步 tokens/s。图执行仅减少主机逐算子调度，未改变特征映射、FP32 状态或注意力数学定义。", "", "## 仅特征映射", "", "以下仅包含一层、全部 10 个头的 Q/K 特征映射，输入为相同真实文本的第一层 RoPE 后 Q/K。没有包括 QKV 投影、状态更新、归一化读出、MLP 主干或词表头。采用两种映射的实际实现；每条件两轮，每轮 21 次。此微基准采用 NPU event 时间中位数，降低单次主机同步固定成本的影响；eager 的 event 间隔仍可能包含主机调度空隙，graph 列更能体现设备执行时间。原始墙钟时间也完整保存。", "", "| Batch | tokens | 执行 | AD μs | Hedgehog μs | HH/AD |", "|---:|---:|---|---:|---:|---:|"]
    for r in summary["features"]:
        lines.append(f"| {r['batch']} | {r['length']} | {r['execution']} | {r['ad64']['median_event_ms']*1000:.2f} | {r['hedgehog']['median_event_ms']*1000:.2f} | {r['hedgehog_over_ad_event_time']:.3f}× |")
    lines += ["", "## 缓存与数值验证", "", "每请求的 12 层 FP32 状态为 `12 × 10 × m × (64 + 2) × 4` 字节，包含 KV 累积矩阵、归一化累积向量和稳定缩放量。AD 为 2,027,520 字节（1.934 MiB），Hedgehog 为 4,055,040 字节（3.867 MiB），AD 恰好减半；两者都不随已读上下文长度增长。Batch=8 分别 15.469 MiB 和 30.938 MiB。此处是注意力持久缓存，未包括模型权重、临时激活或图执行内存池。所有测量都核实解码前后缓存存储字节数相同。", ""]
    for c in bench["correctness"]:
        fp = max(r['max_abs'] for r in c['fp32_cache_vs_full_sequence'])
        bf = max(r['cache_vs_full_sequence']['relative_l2'] for r in c['steps'])
        gr = max(r['graph_vs_eager']['max_abs'] for r in c['steps'])
        lines.append(f"- {c['method']}，batch={c['batch']}：FP32 缓存与原模型整段前向最大绝对误差 {fp:.3g}；BF16 相对 L2 最大 {bf:.3%}；图执行与 eager logits 最大绝对误差 {gr:g}。")
    lines += ["", "验证输入长 73 tokens，先 prefill 65，再逐步读入 8 个不同的真实后续 token；每一步比对原模型全长对应位置 logits，也逐项比对全部层的 KV/z/scale，检查设备位置最终为 73。FP32 误差在 3e-6 内；BF16 的分步/整段差异来自不同矩阵形状的舍入路径。最初以 0.5% 相对 L2 作 BF16 断言时未通过，随后增加独立 FP32 对照，证实缓存代数一致；最终 BF16 上限为 1%，实测最高低于 0.6%。初次日志保留于 work/logs/100m-inference-check.log，没有作为速度样本。", "",
             "## 复现与文件", "", "在项目根目录执行下列命令，只进行推理校验和计时，不启动训练：", "", "```bash", "bash scripts/benchmark_100m_inference.sh", "```", "", "脚本默认创建带 UTC 时间戳的新输出目录，避免覆盖原始结果。也可用第一个位置参数指定新目录，后续传 `--samples`、`--rounds` 等参数。依赖现有 nonlinear-qk 环境及本地留出数据。", "",
             "- benchmark.json：完整模型原始采样、源码/数据/权重哈希、环境、缓存及正确性记录。", "- features.json：特征映射原始采样。", "- summary.json：合并中位数、四分位区间、分轮结果。", "- latency.png / latency.pdf：可分享图表。", "- experiments/inference_100m_runtime.py：新增推理适配器；训练模型源码和训练配置均保持不变。", "", "![Latency comparison](latency.png)", ""]
    (base/"REPORT.zh.md").write_text("\n".join(lines))
    print(base/"REPORT.zh.md")


if __name__ == "__main__": main()
