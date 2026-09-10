"""Build a reviewable preparation report strictly from completed local probes."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from pretrain_100m.common import ROOT,DEFAULT_CONFIG,atomic_json,digest,source_binding


def main():
    c=json.loads(DEFAULT_CONFIG.read_text())
    base=ROOT/"work/pretrain_100m_2b"
    output=base/"validation"
    names={"ad64":"AD m64","softmax":"Softmax","favor64":"FAVOR+ m64","hedgehog":"FLA Hedgehog"}
    ranges={"ad64":[7.,10.],"softmax":[4.,5.],"favor64":[6.,8.],"hedgehog":[8.,10.]}
    rows=[]
    for method in c["methods"]:
        candidates=[]
        for path in (base/"probes").glob(f"throughput_*_{method}_s11/probe_result.json"):
            r=json.loads(path.read_text());b=json.loads((path.parent/"run.json").read_text())["binding"]
            if b["micro_batch"]==c["micro_batch"][method] and b["model_config"]["chunk_size"]==c["architecture"]["chunk_size"] and b["protocol"]["global_batch_sequences"]==c["global_batch_sequences"]:
                candidates.append((path,r,b))
        assert candidates,method
        path,r,b=candidates[-1]
        assert b["source_sha256"]["src/pretrain_100m/model.py"]==digest(ROOT/"src/pretrain_100m/model.py")
        rows.append({"method":method,"parameters":r["parameters"]["total"],"micro_batch_per_npu":c["micro_batch"][method],
                     "gradient_accumulation":c["global_batch_sequences"]//(2*c["micro_batch"][method]),
                     "tokens_per_second":r["measured_tokens_per_second"],"compute_hours_2b":r["estimated_2b_hours_compute"],
                     "planning_hours_min":ranges[method][0],"planning_hours_max":ranges[method][1],
                     "peak_allocated_gib":r["max_allocated_gib"],"peak_reserved_gib":r["max_reserved_gib"],
                     "source":str(path.relative_to(ROOT))})
    with (output/"selected_throughput.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    estimates={"rows":rows,"ideal_compute_hours_per_seed":sum(r["compute_hours_2b"] for r in rows),
               "planning_hours_per_seed":[sum(r[f"planning_hours_{s}"] for r in rows) for s in ["min","max"]],
               "planning_hours_default_queue":[len(c["seeds"])*sum(r[f"planning_hours_{s}"] for r in rows) for s in ["min","max"]],
               "scope":"Real 100M, T2048, two-NPU forward/backward/DDP/optimizer updates; 8 updates/trial, first 3 excluded. Fresh weights; not a promise of late-training speed.",
               "config_sha256":digest(DEFAULT_CONFIG),"source_sha256":source_binding()}
    atomic_json(output/"time_estimate.json",estimates)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    colors=["#2a6fbb","#555555","#3b9272","#c27c36"]
    labels=[names[r["method"]] for r in rows]
    axes[0].bar(labels,[r["tokens_per_second"]/1000 for r in rows],color=colors)
    axes[0].set(ylabel="Thousand tokens / second",title="Measured training throughput (2 x 910B3)")
    axes[1].bar(labels,[r["compute_hours_2b"] for r in rows],color=colors)
    axes[1].set(ylabel="Hours per 2B-token run",title="Compute-only extrapolation; fresh weights")
    for ax in axes: ax.tick_params(axis="x",labelrotation=15);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    fig.savefig(output/"throughput_and_time.png",dpi=180);fig.savefig(output/"throughput_and_time.pdf");plt.close(fig)
    fmap=json.loads((output/"feature_map_speed.json").read_text())["summary_ms"]
    resume=json.loads((output/"resume.json").read_text())
    assert resume["passed"]
    npu=json.loads((output/"npu_components.json").read_text())
    end=json.loads((output/"end_to_end.json").read_text())
    assert end["passed"]
    ad=next(r for r in rows if r["method"]=="ad64")
    hh=next(r for r in rows if r["method"]=="hedgehog")
    text="\n**实测吞吐与训练时间估计**\n\n"
    text+="下表来自上述约 100M 模型、2048 长度和真实训练数据的双卡检查。每个候选运行 8 次更新，排除前 3 次预热，以后 5 次完整前向、反向、梯度累积、HCCL、裁剪与优化器更新计时。\n\n"
    text+="| 方法 | 每卡 microbatch / 累积 | 双卡 tokens/s | 单卡峰值分配显存 | 2B 纯计算外推 | 实际排期预留 |\n|---|---:|---:|---:|---:|---:|\n"
    for r in rows:
        text+=f"| {names[r['method']]} | {r['micro_batch_per_npu']} / {r['gradient_accumulation']} | {r['tokens_per_second']:,.0f} | {r['peak_allocated_gib']:.2f} GiB | {r['compute_hours_2b']:.2f} h | {r['planning_hours_min']:g}–{r['planning_hours_max']:g} h |\n"
    text+="\n四种方法单 seed 纯计算外推合计约 %.1f 小时；考虑评测、检查点、运行波动及后期数值细分，建议按 **25–33 小时/seed，默认两个 seed 合计 50–66 小时（约 2–3 天）** 排期。计时没有覆盖完整 2B 的后期训练状态，尤其 AD 的数值细分次数可能增加，因此这不是保证完成时间。\n"%estimates["ideal_compute_hours_per_seed"]
    text+="\n已比较 AD 每卡 4/8/16/24/32、FAVOR+ 8/16/32、Hedgehog 8/16/24、softmax 16/32/48；AD 32 出现 OOM，其余所列候选通过。64-token chunk 也优于所测 128-token chunk。正式配置选用实测吞吐最佳的候选，不以占满显存为目标；保留的 HBM 有利于后期数值细分。两卡利用率的逐次采样随各 throughput 日志保存；AICore 的高活跃度不等于达到理论 Cube FLOPS。\n"
    text+="\nsoftmax 调度检查确认实际调用 `npu::npu_fusion_attention_v3` / `aclnnFlashAttentionScore` 及其反向算子，而非只根据函数名推断已加速。\n"
    text+="\n**AD 与本次官方 FLA Hedgehog 的速度结论**\n\n"
    text+="同一真实数据骨干产生的 Q/K，B=16、H=10、T=2048、d=64、FP32，排除 QKV/RoPE 准备时间，两轮反转测试顺序：\n\n| 仅 feature map | AD m64 | FLA Hedgehog m128 |\n|---|---:|---:|\n"
    text+=f"| Q/K 前向 | {fmap['ad64']['forward']:.3f} ms | {fmap['fla_hedgehog']['forward']:.3f} ms |\n"
    text+=f"| Q/K 前向+输入及参数梯度 | {fmap['ad64']['forward_backward']:.3f} ms | {fmap['fla_hedgehog']['forward_backward']:.3f} ms |\n"
    text+=f"\n**AD 对这个官方 Hedgehog 没有单独 feature-map 计算速度优势**，本次 forward/backward 反而慢约 {fmap['ad64']['forward_backward']/fmap['fla_hedgehog']['forward_backward']:.2f} 倍。完整训练中 AD 吞吐高约 {(ad['tokens_per_second']/hh['tokens_per_second']-1)*100:.1f}%，是更小特征/状态维度等整条计算路径共同产生的结果，不能归因于 feature map 本身更快，也不能从初期短测保证整个 2B 阶段一直保持相同比例。\n"
    text+="\n**完成的验证与产物位置**\n\n"
    text+="* `validation/contracts.json`：15 组独立目标覆盖检查；两个 seed 的完整 2B 顺序/预算；四方法共同骨干逐字节一致；Hedgehog 官方函数的值和输入/参数梯度一致；三种线性方法对 FP64 稠密因果参考的输出与梯度检查通过。\n"
    text+=f"* `validation/npu_components.json`：四种小模型的实际 NPU BF16 融合路径与 CPU FP32 参考相比，全参数梯度相对 RMS 最大为 {max(v['all_parameter_gradient_relative_rms'] for k,v in npu.items() if k!='fused_adamw_against_cpu'):.3%}；全忽略标签微批量的 loss 和梯度均为 0。融合 AdamW 的 3 步更新与 CPU 参考最大参数误差 {npu['fused_adamw_against_cpu']['max_parameter_error']:.2e}。\n"
    text+=f"* `validation/resume.json`：真实双卡运行，4097 目标的末批场景，连续 3 步与 2 步保存后恢复 1 步的模型和优化器通过严格 FP32 容差检查；共检查 {resume['tensors_compared']} 个张量，最大绝对差 {resume['max_absolute_error']:.2e}。不是逐位确定性保证；同一次运行的两卡模型逐字节相同，步数、目标计数和学习率严格一致。\n"
    text+="* `validation/end_to_end.json`：最终正式结构的短时保存/加载和全部五类评测入口检查通过。这里只证明代码流程可运行，不是 2B 训练质量结果。\n"
    text+="* `validation/time_estimate.json` / `selected_throughput.csv`：所选吞吐、候选来源、时间估计与最终代码/配置绑定；`throughput_and_time.png` / `.pdf` 为静态图。\n"
    text+="\n正式产物启动后写入 `work/pretrain_100m_2b/runs/{method}_s{seed}`，汇总到 `work/pretrain_100m_2b/reports`。目前没有正式 2B 质量指标。\n"
    report=ROOT/"reports/PRETRAIN_100M_2B.zh.md"
    prefix=report.read_text().split("<!-- MEASUREMENTS_AND_VALIDATION -->")[0]
    report.write_text(prefix+"<!-- MEASUREMENTS_AND_VALIDATION -->\n"+text)
    print(json.dumps(estimates,indent=2))


if __name__=="__main__":main()
