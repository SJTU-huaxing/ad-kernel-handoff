"""Build a self-contained Chinese report and exportable research figures."""
import json,math,statistics,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
LABEL={'teacher':'原模型','exact_split':'精确 attention 拆分控制','partition':'新正 kernel','galerkin':'Galerkin','favor_plus':'FAVOR+（m64）','favor_plus_640':'FAVOR+（m640 预算对照）'}

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    ppl=json.loads((P/'results/ppl_optimized.json').read_text())['results']
    summary=json.loads((P/'results/summary.json').read_text())
    ops=json.loads((P/'results/operator_benchmark.json').read_text())
    bench=json.loads((P/'results/model_benchmark_optimized.json').read_text())
    spectrum=json.loads((P/'results/tightened_spectrum.json').read_text())['results']
    checks=json.loads((P/'results/runtime_checks.json').read_text())
    assert len(ppl)==20 and len(bench['results'])==15 and len(spectrum)==16
    pt=[]
    for method in ['teacher','exact_split','partition','galerkin','favor_plus','favor_plus_640']:
        row=[LABEL[method]]
        for ds in ['internal','official']:
            a=[r['perplexity'] for r in ppl if r['method']==method and r['dataset']==ds]
            row.append(f'{a[0]:.6f}' if len(a)==1 else f'{statistics.median(a):.6f} [{min(a):.6f}, {max(a):.6f}]')
        pt.append(row)
    op=[]
    for n in [1,128,2048,8192]:
        vals={r['method']:r['feature_pair']['median_ms'] for r in ops['results'] if r['heads']==4 and r['n']==n and r['optimized'] and r['cuda_graph']}
        op.append([n]+[f'{vals[m]*1000:.2f}' for m in ['favor_plus','partition','galerkin']]+[f"{vals['partition']/vals['favor_plus']:.2f}×"])
    bt=[]
    for n in [1024,4096,8192]:
        for method in ['teacher','exact_split','partition','favor_plus','galerkin']:
            r=next(r for r in bench['results'] if r['prompt_tokens']==n and r['method']==method)
            bt.append([n,LABEL[method],f"{r['prefill']['median_ms']:.3f}",f"{r['decode_ms_per_token']:.3f}",f"{r['decode_tokens_per_second']:.2f}",f"{r['kv_cache_bytes']/2**20:.0f}"])
    st=[]
    for ds in ['internal','official']:
        for r in spectrum:
            if r['dataset']==ds and r['power_iterations']==1:
                b=r['bounds']['64']
                st.append([ds,f"L{r['head'][0]}H{r['head'][1]}",f"[{b['lower']:.6f}, {b['upper']:.6f}]",f"{b['frozen_positive_risk']:.6f}",f"{b['certified_empirical_suboptimality_ratio_at_least']:.2f}×",f"{r['missed_energy_fraction']:.6f}"])
    old=P.parent/'distribution_operator/results'
    projection=[]
    internal=json.loads((old/'internal_summary.json').read_text())[-1]['heads']
    for r in internal:
        b=r['positive']['64'];projection.append([f"L{r['head'][0]}H{r['head'][1]}",f"{b['nmse']:.6f}",f"{b['projection_nmse']:.6f}",f"{b['conditional_mean_estimation_gap']:.6f}"])
    galcounts=[r for r in ppl if r['method']=='galerkin']
    text=f'''本轮结论：按用户选定的范围，在完整冻结 Qwen2.5-1.5B 中替换现有 4/336 个 query head。新正 kernel 的困惑度优于这里复现的五组 FAVOR+，但仍劣于原模型；没有测得端到端推理收益。公平融合和去除主机启动开销后，新 kernel 相对 FAVOR+ 的特征计算差距反而更明显。“接近未知总体 rank-m 理论下界”尚未成立；本轮收紧了完整留出经验算子的谱界，用于进一步检验这一主张。

**实验范围与复现条件。** 模型固定 Qwen2.5-1.5B，revision `8faed761d45a263340a0528343f099c05c9a4323`；L14H0、L14H6、L27H0、L27H6，m=64，head dimension=128。没有训练/微调模型，也没有重新拟合新 kernel。模型使用 BF16，特征及因果状态 FP32，TF32 关闭。GPU 为 RTX 3090。所有计时期间 GPU 上只有这一项测试。原始拟合目标始终为 exp(qᵀk/√128)；数值上的公共尺度/互补特征尺度在 attention 分子与分母中抵消，没有把目标换成归一化分数。

**FLOPs 与时间的关系。** 较大的 GEMM 能提高吞吐，启动和读写开销也会使时间不与 FLOPs 成比例。这与 NVIDIA 的矩阵乘法性能说明一致，但这种现象不是新 kernel 独有的优势。此前 FP32 实现中总 FLOPs 约 12.75 倍、总耗时约 2.72 倍，包含小算子启动及公共聚合开销，不能据此推断算子优化后差距必然缩小。[NVIDIA 官方说明](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)

实际实现了双方的融合算子：新 kernel 将 exp epilogue 融合，用 Triton 树遍历替代稠密路径评分 GEMM，单 token 使用 GEMV+exp；FAVOR+ 融合范数、bias、exp，单 token 融合 GEMV。Galerkin 使用同样的 landmark 优化。双方同时测试 eager 与 CUDA Graph；未使用低精度近似偷偷减少任何一方的工作量。融合主要减少中间读写与启动，无法消除 1024 个 landmark 的数学计算。[Triton 官方融合示例](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)

下表是 **Q/K 特征对的耗时，微秒**，4 heads，FP32，双方已优化，CUDA Graph，30 次重复的中位数；不是完整 attention 或完整模型耗时。

{table(['N','FAVOR+ μs','新正 kernel μs','Galerkin μs','新 / FAVOR+'],op)}

用同一特征计时口径比较 N=2048：本轮未融合 eager 的 FAVOR+/新 kernel 分别为 0.2847/1.1623 ms，约 4.08 倍；融合加 CUDA Graph 后为 0.0676/0.7014 ms，约 10.38 倍。不能将后一特征比值与此前包含公共聚合的总耗时比值直接当成同一个指标比较。

48 组实现比较的最大特征相对 L2 差异为 {max(r['relative_l2'] for r in ops['checks']):.3g}；检查的分区 cell 一致。以 N=2048 为例，新方案的主导特征 GEMM 约 6.442 GFLOPs、FAVOR+ 约 0.268 GFLOPs，按表中时间计算的有效吞吐约为 9.18 / 3.97 TFLOP/s。较大 GEMM 的吞吐优势仍然存在，但这不等于延迟优势，也不是实际 SM occupancy 测量。N=1 的差距更小，不能外推到其他 batch、GPU 或部署引擎。尚未完成联合 GEMM-exp-GEMM 的最优融合或穷尽 autotuning，所以这也不是硬件性能上界；可以确认的是，当前证据不支持“新 kernel 比 FAVOR+ 更适合 GPU，因而能靠写算子抹平计算差距”。

**分区 kernel 的实际稀疏状态。** 令 κ̂(q,k)=B[c(q),d(k)]，B≥0。定义

\\[
S_s(t)=\\sum_{{j\\le t:d(k_j)=s}}v_j,\\qquad n_s(t)=\\sum_{{j\\le t}}1_{{d(k_j)=s}}.
\\]

则

\\[
y_t=\\frac{{\\sum_s B_{{c(q_t),s}}S_s(t)}}{{\\sum_s B_{{c(q_t),s}}n_s(t)}}.
\\]

每一步仅更新一个 key cell 的 value 桶和计数，更新 O(d_v)，读取输出 O(m d_v)。已在 Triton 中实现只写这一桶，并融合读取输出；FAVOR+ 也使用融合的状态更新与读取。该结构优势针对状态更新，1024-landmark 的特征成本仍是主要问题。Prefill 使用并行块状态的前缀和，加上块内精确因果乘积，固定块长 64 时为线性复杂度。

因果验证使用一份真实 1024-token Q/K/V：显式下三角 kernel、分块计算、逐 token 更新及截断前缀互相比较；Float64 最大相对 L2 误差 {max(max(r['dense_relative_l2'],r['recurrent_relative_l2'],r['prefix_relative_l2']) for r in checks):.3g}。最终融合递推 FP32 对 Float64 的最大相对 L2 误差 {max(r['fp32_recurrent_relative_l2'] for r in checks):.3g}。完整 BF16 模型中 prefill 与 cached decode 的最后 token logits 并非逐位相同，原模型本身也存在该差异；五种情形的最后 token top-1 均一致。这只是实现检查，不是生成质量评估。

**完整模型的局部替换困惑度。** 使用全部 256 篇内部留出 WikiText103 文章、20 篇固定 WikiText2 官方 test 文章，各取 1024-token 前缀，分别计 261,888 与 20,460 个 next-token loss。两者都与 kernel 构造文档分离。官方一列是固定子集困惑度，不是标准拼接 WikiText2 benchmark 分数。所有因果位置都参与替换，下游隐藏状态和 Q/K 全部重新计算。五个 FAVOR+ 种子全部报告，没有挑最好的种子。

{table(['方案','内部 256 篇 PPL','官方 test 子集 20 篇 PPL'],pt)}

FAVOR+ m64 单元格为五种子的中位数 [最小值, 最大值]。m640 是预先固定的五个 128-node block 的联合，不是筛选的最好种子，也不是五个独立 m640 实验。它的原有矩形 attention 主导 FLOPs 约为新 m64 kernel 的 78%，因此只是接近计算预算的补充质量对照，不是严格相同 FLOPs；本轮没有对 m640 写对应的全部融合算子或计入完整模型速度表。精确拆分控制组保留相同的 Q-head 拆分方式，但每一组都使用精确 SDPA，用于区分拆分实现造成的数值变化。配对文档 bootstrap 使用 10,000 次重采样；每个 FAVOR+ 种子的“新 kernel NLL − FAVOR+ NLL”区间记录在 `results/summary.json`。这衡量样本文档的不确定性，不是未知总体谱的置信区间。

Galerkin 的完整模型结果没有 NaN loss，但内部/官方分别累计出现 {galcounts[0]['nonpositive_denominators']} / {galcounts[1]['nonpositive_denominators']} 个非正分母。没有加 epsilon、截断负数或替换失效输出。其带符号 kernel 仍存在分母抵消问题，有限 PPL 不能证明所有输入下稳定。FP64 特征/状态的额外 4+4 文档检查及逐文档差异已记录；不能把单块约 1e-6 的算术一致性外推为完整 BF16 模型逐位一致。

**真实推理测试。** 下表包含整个模型、KV cache 更新、最后位置 LM head。Prefill 3 次预热+7 次测量，decode 1 次预热+5 次测量，每次 32 个固定后续 token，batch=1。计时提示词由真实留出文章拼接，4096/8192 长度只用于性能测量，本轮没有验证这些长度的困惑度。实现是 Hugging Face eager model，不是生产服务引擎。

{table(['提示词长度','方案','Prefill ms','Decode ms/token','token/s','KV cache MiB'],bt)}

局部替换无法消除 K/V cache：每层被替换的两个 Q head 分别与另外五个未替换 Q head 共用一个 KV head，因此所有 KV 仍需保留。新方案还增加 132,096 字节的 FP32 线性状态，外加约 6 MiB 的静态 landmark/projection/分区参数。拆分控制也包含选择剩余 Q head、复制对应 K/V 张量的开销。当前 eager 适配器不是最优融合 hybrid attention 算子，速度表不能作为不可加速的数学证明。各 head 成本相近的 Amdahl 估计下，即使所选 head 完全免费，也只消掉 attention 部分的 4/336=1/84，整模型改善空间有限。4/336-head 的结果不能用于预测全 336-head 线性化后的性能或质量；但这次部署范围内，收益主张没有得到支持。[GQA 原论文](https://arxiv.org/abs/2305.13245)

**收紧经验谱界，而非把有限样本称为总体。** 原本固定训练基底给出的区间较宽。本轮对官方全部 10240×10240、内部全部 131072×131072 的经验乘积分布做流式算子乘法，未将 kernel 全矩阵放入显存。随机 range 取 256 维并做一次 subspace iteration，只用于诊断，不改变任何待评估 feature map。[随机子空间方法原论文](https://arxiv.org/abs/0909.4061)

对 n×n 的 kernel 数值矩阵 K 和任意正交列矩阵 U，设 τ_j=s_j(UᵀK)/n，H=||K||²_F/n²。H 是经验乘积分布下的平方能量，E_{{m,emp}}* 也按经验均值定义，则有确定性的经验界

\\[
\\frac{{\\sum_{{j>m}}\\tau_j^2}}{{H}}
\\le E_{{m,\\mathrm{{emp}}}}^\\star/H
\\le 1-\\frac{{\\sum_{{j\\le m}}\\tau_j^2}}{{H}}.
\\]

这里的随机性只影响区间紧度，不影响正交子空间投影所给区间的数学有效性。左界来自投影的奇异值不大于原算子奇异值；右界来自 U(UᵀK)_m 的可行 rank-m 近似。区间宽度恰为漏掉的平方能量。全部运算 Float64，重新积分的 H 与既有穷举结果核对，正交误差检查通过。

{table(['经验分布','head','rank-64 最优相对误差区间','新正 kernel 误差','R / E* 至少','区间宽度'],st)}

“R/E* 至少”用 R 除以 E* 的**上界**得到，方向不能反过来；它用于证明与经验无约束 rank-64 最优解存在多大差距。它不是正特征最优误差 E_m⁺ 的界，也不是未知总体的倍数证书。

分区模型还有更直接的误差分解：固定 Q/K 分区下，条件均值 B* 是 L2 投影；估计的 B̂ 满足

\\[
R(B̂)=R(B^*)+\\sum_{{r,s}}p_rq_s(B̂_{{rs}}-B^*_{{rs}})^2.
\\]

内部完整经验分布对应结果如下；后两列之和为第一列，说明目前主要损失来自分区函数空间，而非仅仅条件均值样本数不足。

{table(['head','冻结新 kernel 风险','固定分区投影误差','B 估计误差'],projection)}

**总体主张需要什么。** Schmidt 尾谱定理在 κ∈L2(P_Q×P_K) 下成立；这项数学结论与经验估计是否可信是两件事。要认证 R_P(κ̂)≤(1+ε)E_m*(P)，至少要给出同一总体下的风险上置信界 U_R 与谱下置信界 L_*，再检验 U_R≤(1+ε)L_*。当前没有这样的证书。即使 131072² 个 kernel pair 都算完，也只有 256 篇留出文档，不能把 pair 数当成独立样本量。

举例：若 Q 与 K 来自相互独立的 n_Q、n_K 篇文档，且逐文档交叉平均损失落在 [0,L]，有界差分能给出误差量级 L√((1/n_Q+1/n_K)log(2/δ)/2)。文档内 token 可相关。应用到核平方能量需要已知的全分布 κ² 上界，应用到谱还需要投影矩、Gram 矩阵及其扰动控制。样本最大值不是合法的全分布上界；重尾下该证书可能十分宽甚至无实际用途。原模型的有界投影可以保证存在有限数学上界，并不保证这个上界足够紧。

总体 Schmidt 定理并不要求 Gaussian；本轮经验谱收紧也没有验证“只用 Q/K covariance 即可预测真实谱”的 Gaussian 外推。此前非 Gaussian 尾部及少数文档主导平方能量的问题仍然存在。

原始 kernel 的 L2 风险与模型质量也不等价。对一个上下文，若 κ、κ̂≥0，Z=Σ_jκ_j，κ̂ 的分母非零，且 ||v_j||≤V，则归一化输出误差满足 ||y−ŷ||≤2V Σ_j|κ_j−κ̂_j|/Z。这个不等式解释了为何仍需逐行相对误差及真实上下文测试；它没有改变 raw-kernel 拟合目标。P_Q×P_K 的边缘乘积分布还不同于真实同文档、带位置及因果限制的 joint distribution。对有符号 Galerkin，正性所提供的这一分母稳定性保障不再自动适用。

因此，本轮支持“该冻结正 kernel 在这些未见文档上较五组 m64 FAVOR+ 和预先固定的 m640 对照保留了更多模型质量”；不支持“已经带来推理收益”或“已经接近真实总体理论下界”。下一步更值得研究的是以更低成本逼近有效谱子空间，并在新文档上控制该子空间的误差；单纯继续打磨 1024-landmark 实现，无法保证消除特征成本。

结果文件：`results/ppl_optimized.json`、`results/summary.json`、`results/operator_benchmark.json`、`results/model_benchmark_optimized.json`、`results/tightened_spectrum.json`、`results/runtime_checks.json`。`ppl.json`、`model_benchmark.json` 保留早期实现结果，主表采用最终版本。

复现命令：使用 `/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python` 执行 `verify_runtime.py`、`evaluate_model.py --output ppl_optimized.json`、`benchmark_operators.py`、`benchmark_model.py`、`tighten_spectrum.py`；使用带 NumPy/Matplotlib 的默认 `python` 执行 `summarize.py`、`write_report.py`。GPU 性能测试必须串行进行。
'''
    (P/'REPORT.zh.md').write_text(text)
    figdir=P/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'partition':'#157f78','favor_plus':'#b56a28','galerkin':'#8060ad','exact_split':'#777777'}
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for ds,x in [('internal',0),('official',1)]:
        baseline=next(r['perplexity'] for r in ppl if r['dataset']==ds and r['method']=='teacher')
        for j,method in enumerate(['exact_split','partition','galerkin','favor_plus']):
            values=[100*(r['perplexity']/baseline-1) for r in ppl if r['dataset']==ds and r['method']==method]
            center=np.median(values);xx=x+(j-1.5)*.18
            axs[0].bar(xx,center,width=.17,color=colors[method],label=method if x==0 else None)
            if len(values)>1:axs[0].errorbar(xx,center,yerr=[[center-min(values)],[max(values)-center]],color='black',capsize=3)
    axs[0].set_xticks([0,1],['Internal 256 docs','Official subset 20 docs']);axs[0].set_ylabel('PPL increase over original (%)');axs[0].legend(fontsize=8)
    for method in ['favor_plus','partition','galerkin']:
        rr=sorted([r for r in ops['results'] if r['heads']==4 and r['optimized'] and r['cuda_graph'] and r['method']==method],key=lambda r:r['n'])
        axs[1].loglog([r['n'] for r in rr],[r['feature_pair']['median_ms']*1000 for r in rr],'o-',color=colors[method],label=method)
    axs[1].set_xlabel('Tokens N');axs[1].set_ylabel('Q/K feature latency (μs)');axs[1].legend(fontsize=8)
    for method in ['partition','favor_plus','galerkin','exact_split']:
        rr=sorted([r for r in bench['results'] if r['method']==method],key=lambda r:r['prompt_tokens'])
        ratios=[]
        for r in rr:
            base=next(s['decode_ms_per_token'] for s in bench['results'] if s['method']=='teacher' and s['prompt_tokens']==r['prompt_tokens'])
            ratios.append(r['decode_ms_per_token']/base)
        axs[2].plot([r['prompt_tokens'] for r in rr],ratios,'o-',color=colors[method],label=method)
    axs[2].axhline(1,color='black',linestyle='--');axs[2].set_xlabel('Prompt tokens');axs[2].set_ylabel('Full-model decode latency / original');axs[2].legend(fontsize=8)
    fig.suptitle('Frozen Qwen2.5-1.5B: four of 336 query heads replaced (m=64)',fontsize=12)
    fig.tight_layout();fig.savefig(figdir/'deployment.png',dpi=180);fig.savefig(figdir/'deployment.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,ds in zip(axs,['internal','official']):
        rr=[r for r in spectrum if r['dataset']==ds and r['power_iterations']==1]
        for i,r in enumerate(rr):
            b=r['bounds']['64'];ax.vlines(i,b['lower'],b['upper'],color='#2457a7',linewidth=5)
            ax.scatter(i,b['upper'],s=40,color='#2457a7');ax.scatter(i,b['frozen_positive_risk'],marker='x',s=80,color=colors['partition'])
        ax.set_xticks(range(4),[f"L{r['head'][0]}H{r['head'][1]}" for r in rr]);ax.set_title(ds+' empirical product')
        ax.set_ylabel('Relative raw-kernel squared error');ax.set_ylim(bottom=0)
    axs[0].plot([],[],color='#2457a7',linewidth=4,label='Certified empirical rank-64 bracket');axs[0].scatter([],[],marker='x',color=colors['partition'],label='Frozen positive kernel');axs[0].legend(fontsize=8)
    axs[0].set_ylim(0,max(r['bounds']['64']['frozen_positive_risk'] for r in spectrum)*1.1)
    fig.suptitle('Full empirical operator certificates — not population confidence bounds');fig.tight_layout()
    fig.savefig(figdir/'spectral_gap.png',dpi=180);fig.savefig(figdir/'spectral_gap.pdf');plt.close(fig)
    print('Report and figures written.',len(text))

if __name__=='__main__':main()
