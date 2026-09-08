"""Scientific figures and a Chinese interpretation of the checkpoint diagnostics."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent;D=P/'diagnostics'
COLORS=dict(mlp='#4477aa',kan='#ee7733',mulkan='#228833')
NAMES=dict(mlp='MLP',kan='KAN',mulkan='mulKAN')


def table(columns,rows):
    return '\n'.join(['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    r=json.loads((D/'convergence_tail.json').read_text())
    assert len(r['stratified_prediction'])==18 and len(r['dense_blocks'])==32
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,loss,title in zip(axes,['logcosh','poisson'],['Fixed validation: log-cosh','Fixed validation: raw I-divergence / mass']):
        for c in r['history']:
            if c['loss']!=loss:continue
            rows=[a for a in c['curve'] if a['step']>=512]
            ax.errorbar([a['step'] for a in rows],[a['mean'] for a in rows],yerr=[a['seed_sd'] for a in rows],
                        color=COLORS[c['method']],marker='o',label=NAMES[c['method']],capsize=3)
        ax.set_xscale('log',base=2);ax.set_xticks([512,1024,2048,4096],labels=[512,1024,2048,4096])
        ax.set_xlabel('Updates / fresh training documents');ax.set_title(title);ax.grid(alpha=.2)
        ax.axvspan(2048,4096,alpha=.06,color='gray')
    axes[0].legend();fig.suptitle('Same 32 validation documents throughout | mean ± SD across 3 seeds')
    for ext in ['png','pdf']:fig.savefig(D/f'late_convergence.{ext}',dpi=160)
    plt.close(fig)
    paired=r['paired_tail'];x=np.arange(4);labels=[f"L{a['head'][0]} H{a['head'][1]}" for a in paired]
    dense=[]
    for i in range(4):
        rows=[a['per_head'][i] for a in r['dense_blocks']]
        dense.append(dict(head=paired[i]['head'],first4_energy_share=sum(a['first4_energy'] for a in rows)/sum(a['energy'] for a in rows),
                          first4_average_attention=float(np.mean([a['first4_mean_row_attention'] for a in rows]))))
    fig,axes=plt.subplots(1,2,figsize=(10,3.9),layout='constrained')
    axes[0].bar(x-.18,[100*a['tail_energy_share'] for a in paired],.36,label='Share of sum(K²)',color='#4477aa')
    axes[0].bar(x+.18,[100*a['tail_mass_share'] for a in paired],.36,label='Share of sum(K)',color='#ee7733')
    axes[0].set_title('Largest ~0.1% kernel pairs (256 documents)')
    axes[1].bar(x-.18,[100*a['first4_energy_share'] for a in dense],.36,label='Global share of sum(K²)',color='#4477aa')
    axes[1].bar(x+.18,[100*a['first4_average_attention'] for a in dense],.36,label='Mean row attention mass',color='#228833')
    axes[1].set_title('First 4 keys (32 complete rectangular blocks)')
    for ax in axes:
        ax.set_xticks(x,labels=labels);ax.set_ylabel('Percent');ax.set_ylim(0,105);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    for ext in ['png','pdf']:fig.savefig(D/f'tail_vs_attention.{ext}',dpi=160)
    plt.close(fig)
    curve_rows=[]
    for a in r['history']:
        curve_rows.append([a['loss'],NAMES[a['method']],f"{a['curve'][-2]['mean']:.4f}",f"{a['curve'][-1]['mean']:.4f}",f"{100*a['relative_drop_2048_4096']:.1f}%"])
    tail_rows=[[str(a['head']),f"{100*a['tail_energy_share']:.2f}%",f"{100*a['tail_mass_share']:.2f}%",f"{100*a['top_single_pair_energy_share']:.2f}%",f"{a['energy_weight_concentration_count']:.2f}"] for a in paired]
    dense_rows=[[str(a['head']),f"{100*a['first4_energy_share']:.2f}%",f"{100*a['first4_average_attention']:.2f}%"] for a in dense]
    pred_rows=[]
    for loss in ['logcosh','poisson']:
        for method in ['mlp','kan','mulkan']:
            rows=[a for a in r['stratified_prediction'] if a['loss']==loss and a['method']==method]
            for h in range(4):
                vals=np.mean([[a['per_head'][h][f] for f in ['tail_sse_share','tail_median_prediction_ratio','tail_mass_prediction_ratio']] for a in rows],0)
                pred_rows.append([loss,NAMES[method],str(paired[h]['head']),f'{100*vals[0]:.2f}%',f'{vals[1]:.4f}',f'{vals[2]:.4f}'])
    text=r'''有训练尚不充分的证据，但现有日志不能把优化不足与表达能力不足完全分开。另一方面，“0.1% 贡献大部分平方能量”揭示的是目标风险的极端权重集中，不等于 attention 概率集中，也不能直接推断幂律或无穷方差。本次只读取已有数据和 checkpoint 做诊断，没有追加梯度训练或变更原实验。

**末段损失是否仍在下降**

必须比较同一个集合上的风险。旧日志的 `loss` 是当前一篇文档的一次 loss，不是固定训练集或移动平均；I-divergence 训练还省略了随目标变化的常数，跨文档的负数变化更不能作为收敛证据。下面使用全程固定的前 32 篇验证文档，模型均为 N=4096、小参数预算，先平均固定四个 head，再平均三个 seed：

'''+table(['训练目标','模型','2048步','4096步','下降比例'],curve_rows)+r'''

后半程仍有 5.1%—10.3% 改善，不能把当前 checkpoint 当作已达到该结构的最佳逼近性能。准确说法是：存在优化尚未充分完成的迹象；已有稀疏记录没有证明最终几十步还在下降，也没有证明延长训练一定继续改善。学习率已按 4096 步从 0.002 退火至 0.0002，曲线变缓可能包含学习率效应。

不同 head 并不一致。例如 KAN 的 (14,0) 验证 I-divergence/质量从 0.8031 升至 0.8198，而其他 head 带动整体下降。两倍参数的 log-cosh 实验也降低了训练和测试风险：MLP 测试 0.5858→0.5348，KAN 0.8360→0.7698，mulKAN 0.7794→0.6992。这说明容量或参数化可能有影响，但优化轨迹也改变了，不能单独据此识别纯容量瓶颈。

当前结论只能是固定数据、步数、学习率、参数预算下的比较，不能写成充分收敛后的结构上限比较。训练不足与尾部泛化差距可以同时存在；前轮 mulKAN 的 I-divergence 训练 raw NMSE=0.8021、测试=0.8845，不因整体曲线仍下降就消失。

![末段验证曲线](late_convergence.png)

**0.1% 平方能量的精确定义**

每个 head 在 256 篇内部测试文档上有 N=131,072 个配对样本。将 K 从大到小排序，取最大的 ceil(0.001N)=132 个，记集合 S。所报数字是

$$C_2=\frac{\sum_{i\in S}K_i^2}{\sum_{i=1}^{N}K_i^2}.$$

“能量”只是平方和的线性代数术语。C2 不是概率质量，不是模型误差，不表示 99.9% 的 token 没有作用。与之并列的 C1=Σ_SK/ΣK 是整个样本池中的未归一化 kernel 质量占比，也不等于逐 query 归一化后再平均的 attention 概率。

'''+table(['head','最大0.1%的平方和占比 C2','同一集合的质量占比 C1','单个最大样本的平方和占比','平方权重集中度数量'],tail_rows)+r'''

最后一列是 (ΣK²)²/ΣK⁴，即归一化平方权重的 inverse participation ratio；它量化权重集中度，**不是统计上独立样本的数量**。例如 (14,6) 单个样本就占 62.84% 的平方和，前三个占约 98.11%。因此该经验平方风险很容易被少量文档中的极端对主导，测试排名应同时看文档 bootstrap、其他语料及分层误差。

一个简单例子：若 99.9% 的 K=1，0.1% 的 K=100，那么 C2=10/(10+0.999)=90.92%；大值改成 600，C2=360/(360+0.999)=99.72%。指数函数只需将 logits 拉开 log(100)=4.61 或 log(600)=6.40，就能产生这种比例。这个现象不要求数值溢出，也不要求幂律分布或无穷二阶矩。真实有限权重 LLM 的 Q/K 有界结论仍适用。

**平方风险、乘性误差、attention 形状在强调什么**

令 δ_i=Khat_i/K_i−1，则

$$\operatorname{NMSE}=\sum_i w_i\delta_i^2,\qquad w_i=\frac{K_i^2}{\sum_jK_j^2}.$$

这是乘性误差的极不均匀加权平均。若普通 99.9% 样本全部拟合正确、仅将尾部预测成接近零，则 NMSE≈C2，可以达到 0.91—0.997。反过来只拟合尾部、把其余样本预测成接近零，也可能得到 NMSE≈1−C2。故 raw NMSE≈1 不意味着所有样本都没学会，较低 raw NMSE 也不意味着多数样本拟合良好。目标能量与误差占比理论上不同；本次追加推断确认，各模型确实主要在尾部留下平方误差，见下表。

令 r=log(Khat/K)，三个目标可以统一写成

$$L_{\rm square}=K^2(e^r-1)^2,\quad L_I=K(e^r-1-r),\quad L_{\rm lc}=\log\cosh r.$$

在小误差附近，它们分别近似 K²r²、Kr²/2、r²/2。对 log-prediction 的梯度分别为 2K²e^r(e^r−1)、K(e^r−1)、tanh r。尤其严重低估时 r→−∞，平方损失对 log-prediction 的梯度趋零，而 I-divergence 的梯度趋 −K。该公式说明 I-divergence 对严重低估的大值具有不同的纠正机制；实际参数梯度还乘以网络 Jacobian，不能据此保证优化必然成功。

训练仍直接拟合原始正 kernel；使用 log-ratio 计算损失或对公共幅度校准，不会把训练标签改成归一化 attention。给每行 K 乘一个仅依赖 query 的公共因子，会改变原始 kernel 风险，而在 linear attention 分母中抵消。这说明两个目标本来就不完全相同，而不是要求放弃用户指定的原始 kernel 拟合目标。

**大值来自哪里：位置、范数和方向**

在 (14,0)/(14,6) 的 132 个最大配对样本中，分别有 107/91 个来自 key 位置 0；但这两个 head 最大的三个平方能量样本均来自非开头位置。例如 (14,6) 最大样本来自内部测试文档 234，q位置1022、k位置260，logit=7.1698。其 q范数42.42（全体均值23.42）、k范数18.48（均值18.41）、余弦0.103（全体均值−0.227）。不能只检查 key 范数，也不能只检查最初几个 token。

第 27 层两个 head 的大值位置更分散，其尾部平均余弦分别约 0.362/0.461，全体约 0.045/0.161。不同 head 同时涉及范数和方向变化，观察到相关性并不等于已经识别因果机制。

在前 32 篇文档的完整合法 512×512 块上，比较前四个 key：

'''+table(['head','前4个key的全局平方和占比','前4个key的平均逐行attention质量'],dense_rows)+r'''

第14层的前几个 key 吸收了大量平均 attention 概率，这与 [StreamingLLM 的 attention sink 观察](https://arxiv.org/abs/2309.17453)相符；但该局部证据不是完整的机制证明，不能把全部 raw-kernel 尾部等同于 sink。上述概率只对前512个 key 归一化、query为后512位置，属于本实验的矩形块诊断，不是完整 causal 上下文的概率。

![平方能量、kernel质量与attention的差异](tail_vs_attention.png)

**已训练模型如何拟合尾部**

下表都是三个 seed 的平均。中位预测比例是先在各 seed 的尾部取 Khat/K 中位数，再平均；尾部总质量比例是 Σ_SKhat/Σ_SK。

'''+table(['损失','模型','head','尾部占总平方误差','尾部中位预测比例','尾部总质量预测比例'],pred_rows)+r'''

例如 MLP/log-cosh 的尾部中位预测只达到目标的约1.5%—4.6%，普通样本的稳健拟合确实伴随大值低估。MLP/I-divergence 在 (14,0)/(14,6) 的中位比例提升到约92%/82%，但 (14,6) 的尾部总质量只恢复约19%，意味着最极端的少数点仍明显低估。不能仅靠一个中位数判断尾部已学好。

**对下一轮训练和理论的启发**

1. 先测收敛再判断结构。保留原始 kernel 标签和每个训练对只用一次的约束，使用更长的新数据流，增加记录频率，固定验证/训练探针，保存中间 checkpoint；学习率先保持可学习区间，再退火。用共同的更长学习率计划比较4096/8192/16384步，而不是分别在每个终点提前退火。另做固定唯一数据对数、改变batch大小的步数对照，可帮助区分数据覆盖与更新步数，但batch变化本身仍有混杂。单遍约束下无法把固定数据重复优化作为完全等价的消融。
2. 关注尾部覆盖，而不只计 token 总数。按训练数据中确定的 logit 桶、key位置、q/k范数和方向模式做分层监控，比较普通样本、sink样本和非sink极端样本。若训练时改变采样分布，应记录采样概率；目标保持原风险时使用正确的逆概率权重，并保留均匀探索。直接按每行完整 QK 找top-k会增加训练成本，不能宣称推断时免费得到该信息。
3. I-divergence 可作为主要损失，辅以较小的 log-cosh 项来兼顾普通样本。权重使用训练集公共尺度校准、验证集选择，不进行逐行标签归一化。这个组合尚未在本次运行中测试，不能宣称已经优于现方案。不要把删去极端值后的好指标当作解决原始 kernel 拟合问题。
4. 正特征可考虑单侧幅度与形状分解：Khat=a(q)b(k)Σ_r f_r(q)g_r(k)，a,b>0。这仍可吸收到m维特征中，保持 linear attention 结构；query幅度在分母抵消，但若目标是原始K，训练时仍需恢复正确幅度。该分解可能改善优化条件，不构成 KAN 特有的表达优势，也未在此次诊断中验证。
5. 理论应说明风险对数据分布的重新加权：raw NMSE 对相对误差使用 dP2/dP=K²/E[K²]；质量归一化 I-divergence 使用 dP1/dP=K/E[K]。在普通样本、sink和非sink极端样本的混合分布上分析谱与逼近，比无条件套用一个高斯更贴近当前证据。经验权重高度集中会使矩估计与经验谱对稀有事件敏感；需要跨文档稳定性和总体泛化分析。

这些结果不证明 m=64 已达到秩瓶颈，也不证明没有秩瓶颈。逐测试块的有符号 SVD oracle 不受正因子、跨文档共享函数、训练估计和优化的限制，不能用其低残差宣布当前网络只缺训练步数。对 KAN/mulKAN 的有价值主张，应是相同预算下更快或更稳地学会这些结构，而不是仅凭通用逼近或乘法节点断言更强。

诊断输入为 `../fits/`、`../data/`；代码为 `../diagnose_convergence_tail.py`、`../summarize_diagnostics.py`；所有数值保存在 `convergence_tail.json`。原始45个训练 checkpoint 未修改。
'''
    # Mathematical strings use a single TeX backslash in Markdown.
    text=text.replace('\\\\','\\')
    (D/'REPORT.zh.md').write_text(text)
    print(table(['损失','模型','2048','4096','下降'],curve_rows))
    print('Wrote',D/'REPORT.zh.md')


if __name__=='__main__':main()
