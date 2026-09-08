"""Aggregate seeds and document-cluster comparisons; render report and figures."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
OUT=P/'results';OUT.mkdir(exist_ok=True)
METHODS=['mlp','kan','mulkan'];DISPLAY=dict(mlp='MLP',kan='KAN',mulkan='mulKAN')
COLORS=dict(mlp='#4477aa',kan='#ee7733',mulkan='#228833')


def markdown_table(columns,rows):
    return '\n'.join(['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    runs=[json.loads(f.read_text()) for f in sorted((P/'fits').glob('*.json'))]
    assert len(runs)==45
    groups=defaultdict(list)
    for r in runs:groups[(r['loss'],r['training_documents'],r['budget_scale'],r['method'])].append(r)
    stats={};long=[]
    for key,rs in groups.items():
        assert len(rs)==3
        s={}
        for split in ['train','validation','test','test_block']:
            for metric,values in rs[0][split]['summary'].items():
                if not isinstance(values,list):continue
                x=np.array([r[split]['summary'][metric] for r in rs])
                s[f'{split}/{metric}']=dict(mean=float(x.mean()),seed_sd=float(x.mean(1).std(ddof=1)),
                    per_head_mean=x.mean(0).tolist(),per_head_seed_sd=x.std(0,ddof=1).tolist())
                for r in rs:
                    for label,value in zip(r['head_labels'],r[split]['summary'][metric]):
                        long.append(dict(loss=key[0],training_documents=key[1],budget_scale=key[2],method=key[3],
                            seed=r['seed'],parameters=r['parameters_per_qk_pair'],split=split,metric=metric,
                            layer=label[0],head=label[1],value=value))
        for field in ['training_seconds','feature_512_pairs_ms']:
            x=np.array([r[field] for r in rs]);s[field]=dict(mean=float(x.mean()),seed_sd=float(x.std(ddof=1)))
        stats[key]=s
    with (OUT/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(long[0]));writer.writeheader();writer.writerows(long)

    # Resample whole documents jointly for two methods, then the three shared seeds.
    # Heads stay fixed: no claim of inference to arbitrary LLM heads.
    boot=[];rng=np.random.default_rng(20260908);repeats=2000
    for loss in ['logcosh','poisson']:
        for method in ['kan','mulkan']:
            a=sorted(groups[(loss,4096,1,method)],key=lambda r:r['seed'])
            b=sorted(groups[(loss,4096,1,'mlp')],key=lambda r:r['seed'])
            for split,metric in [('test','logcosh'),('test','raw_nmse'),('test','normalized_idiv'),('test_block','output_nmse')]:
                def values(rs,field):return np.array([[d[field] for d in r[split]['documents']] for r in rs])
                n=len(a[0][split]['documents']);di=rng.integers(n,size=(repeats,n));si=rng.integers(3,size=(repeats,3))
                if metric in ['raw_nmse','normalized_idiv']:
                    num,den=('raw_sse','raw_energy') if metric=='raw_nmse' else ('idiv','target_mass')
                    xa,xb,da,db=values(a,num),values(b,num),values(a,den),values(b,den)
                    da=da[:,di].sum(2);db=db[:,di].sum(2)
                    delta=(xa[:,di].sum(2)/da-xb[:,di].sum(2)/db).mean(-1).T
                else:delta=(values(a,metric)[:,di]-values(b,metric)[:,di]).mean((2,3)).T
                draws=delta[np.arange(repeats)[:,None],si].mean(1)
                field=f'{split}/{metric}'
                boot.append(dict(loss=loss,method_minus_mlp=method,split=split,metric=metric,
                    estimate=stats[(loss,4096,1,method)][field]['mean']-stats[(loss,4096,1,'mlp')][field]['mean'],
                    percentile_95_ci=np.quantile(draws,[.025,.975]).tolist(),resamples=repeats,
                    note='Paired document+seed bootstrap; only 3 seeds, fixed 4 heads; exploratory interval.'))

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained');ns=[64,512,4096]
    for method in METHODS:
        col=COLORS[method]
        for split,style in [('train','--'),('test','-')]:
            axes[0,0].plot(ns,[stats[('logcosh',n,1,method)][f'{split}/logcosh']['mean'] for n in ns],
                           style,marker='o',color=col,label=f'{DISPLAY[method]} {split}')
        gap=[stats[('logcosh',n,1,method)]['test/logcosh']['mean']-stats[('logcosh',n,1,method)]['train/logcosh']['mean'] for n in ns]
        axes[0,1].plot(ns,gap,'o-',color=col,label=DISPLAY[method])
        for ax,key in [(axes[1,0],'test/raw_nmse'),(axes[1,1],'test_block/output_nmse')]:
            ax.errorbar(ns,[stats[('logcosh',n,1,method)][key]['mean'] for n in ns],
                        yerr=[stats[('logcosh',n,1,method)][key]['seed_sd'] for n in ns],color=col,marker='o',label=DISPLAY[method])
    for ax,title in zip(axes.flat,['Log-cosh: train and held-out test','Test minus train log-cosh','Raw kernel NMSE (evaluation only)','Attention output NMSE (32 documents)']):
        ax.set_title(title);ax.set_xscale('log',base=2);ax.set_xticks(ns,labels=ns);ax.set_xlabel('Training documents / optimizer steps');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,ncol=2);axes[0,1].axhline(0,color='gray',linewidth=.7)
    fig.suptitle('Qwen2.5-1.5B | one pass | m=64 | exactly 73,856 parameters per Q/K pair')
    for ext in ['png','pdf']:fig.savefig(OUT/f'data_scaling.{ext}')
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,key,title in zip(axes,['test/raw_nmse','test/normalized_idiv','test_block/output_nmse'],
                            ['Raw kernel NMSE','Raw I-divergence / target mass','Attention output NMSE']):
        for j,method in enumerate(METHODS):
            ss=[stats[(loss,4096,1,method)][key] for loss in ['logcosh','poisson']]
            ax.bar(np.arange(2)+(j-1)*.23,[s['mean'] for s in ss],.23,yerr=[s['seed_sd'] for s in ss],
                   color=COLORS[method],label=DISPLAY[method],capsize=3)
        ax.set_xticks([0,1],['log-cosh','raw I-divergence']);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=9);fig.suptitle('4,096 training documents | matched parameters | mean ± seed SD')
    for ext in ['png','pdf']:fig.savefig(OUT/f'loss_comparison.{ext}')
    plt.close(fig)

    replacement=json.loads((P/'replacement.json').read_text()) if (P/'replacement.json').exists() else None
    distribution=json.loads((P/'distribution.json').read_text());audit=json.loads((P/'audit.json').read_text())
    serialized=[dict(loss=k[0],training_documents=k[1],budget_scale=k[2],method=k[3],metrics=v) for k,v in stats.items()]
    (OUT/'summary.json').write_text(json.dumps(dict(groups=serialized,paired_bootstrap=boot,audit=audit),indent=2))
    def val(key,field,sd=False):
        s=stats[key][field];return f"{s['mean']:.4f}"+(f" ± {s['seed_sd']:.4f}" if sd else '')
    main_rows=[]
    for loss in ['logcosh','poisson']:
        for method in METHODS:
            key=(loss,4096,1,method)
            main_rows.append([loss if loss=='logcosh' else '原始 I-divergence',DISPLAY[method]]+
                [val(key,f,True) for f in ['test/logcosh','test/raw_nmse','test/normalized_idiv','test_block/output_nmse']])
    size_rows=[]
    for n in ns:
        for method in METHODS:
            key=('logcosh',n,1,method)
            size_rows.append([n,DISPLAY[method],val(key,'train/logcosh'),val(key,'test/logcosh'),
                f"{stats[key]['test/logcosh']['mean']-stats[key]['train/logcosh']['mean']:.4f}",val(key,'test_block/output_nmse')])
    tail_gap_rows=[]
    for method in METHODS:
        key=('poisson',4096,1,method)
        tail_gap_rows.append([DISPLAY[method]]+[val(key,f) for f in ['train/raw_nmse','test/raw_nmse','train/normalized_idiv','test/normalized_idiv']])
    budget_rows=[]
    for budget in [1,2]:
        for method in METHODS:
            key=('logcosh',4096,budget,method)
            budget_rows.append([73856 if budget==1 else 147584,DISPLAY[method]]+
                [val(key,f,True) for f in ['test/logcosh','test/raw_nmse','test_block/output_nmse']]+[val(key,'feature_512_pairs_ms')])
    tail_rows=[[str(r['head']),f"{r['logit_quantiles']['max']:.2f}",f"{100*r['top_0.001_mass_share']:.2f}%",f"{100*r['top_0.001_energy_share']:.2f}%"] for r in distribution['test']]
    floor_rows=[[str(label),f"{distribution['svd']['pooled_relative_floor']['64'][i]:.3e}",
        *[f"{stats[('poisson',4096,1,m)]['test_block/raw_nmse']['per_head_mean'][i]:.4f}" for m in METHODS]]
        for i,label in enumerate(runs[0]['head_labels'])]
    head_rows=[]
    for loss in ['logcosh','poisson']:
        for i,label in enumerate(runs[0]['head_labels']):
            for method in METHODS:
                s=stats[(loss,4096,1,method)]
                head_rows.append([loss,str(label),DISPLAY[method]]+[f"{s[f]['per_head_mean'][i]:.4f}" for f in ['test/raw_nmse','test/normalized_idiv','test_block/output_nmse']])
    ci_rows=[[r['loss'],DISPLAY[r['method_minus_mlp']],r['metric'],f"{r['estimate']:.4f}",
              f"[{r['percentile_95_ci'][0]:.4f}, {r['percentile_95_ci'][1]:.4f}]"] for r in boot]
    replacement_text='完整模型替换评估尚未完成；此处暂不报告困惑度。'
    if replacement and len(replacement['results'])==19:
        teacher=replacement['results'][0];rr=[['原始模型','—',f"{teacher['perplexity']:.4f}",'0.0000',f"{teacher['last512_perplexity']:.4f}"]]
        for loss in ['logcosh','poisson']:
            for method in METHODS:
                rs=[r for r in replacement['results'] if r['scenario'].startswith(f'{loss}_{method}_')]
                pp=np.array([r['perplexity'] for r in rs]);tail=np.array([r['last512_perplexity'] for r in rs])
                delta=np.mean([r['nll']-teacher['nll'] for r in rs])
                rr.append([loss,DISPLAY[method],f'{pp.mean():.4f} ± {pp.std(ddof=1):.4f}',f'{delta:.5f}',f'{tail.mean():.4f}'])
        replacement_text=markdown_table(['损失','模型','困惑度 ± seed SD','相对原模型 ΔNLL','后512位置困惑度'],rr)+'''

这是 32 篇内部测试文章、每篇 1024 token 的局部替换质量实验，只替换第 14/27 层的 0/6 号 head，共 4/336 个 query head；其余保持原模型。不微调，后续隐藏状态和 Q/K/V 重新计算。所有 causal 位置均替换，包含训练未覆盖的前半段 query 和后半段 key，因此包含位置分布外推。困惑度不能当作官方 WikiText benchmark，也不能据此推断全模型线性化后的结果。

使用真实 causal prefix-sum 公式；特征网络及累加用 Float64。逐特征的正对角重标定与逐 query 的公共缩放在分子分母精确抵消，不剪裁分母、不改变训练目标。为保留其他 head，原始 SDPA 仍会计算，因此此实验只评估质量，不评估加速。'''
    report=r'''这轮实验已完成两层 KAN、两层 mulKAN 与两层 MLP 的严格等参数、单遍训练比较。结果没有支持“加入 KAN 或乘法就会全面超过 MLP”。数据增多有帮助，损失函数会改变结构排序；在更直接控制原始 kernel 质量的 I-divergence 对照中，应以表中的误差及完整模型替换结果判断。

**实验对象与公平性**

冻结 Qwen2.5-1.5B，使用真实投影、RoPE 后的 Q/K，head dimension=128，正确按 GQA 将 query head 映射到 KV head。比较第 14/27 层的 0/6 号 head。目标始终是

$$K(q,k)=\exp(q^\top k/\sqrt{128}),\qquad \widehat K=e^{c_h}\sum_{r=1}^{64}f_r(q)g_r(k),$$

其中两支网络独立，$f,g$ 为 softplus 正特征；$1/\sqrt{128}$ 是该 LLM 原有 attention scale，可吸收到 q、k 中。$c_h$ 是只用训练集计算的每 head 公共常数，所有 query 共用，未做逐行标签归一化。特征的固定缩放可同样吸收到网络中，不改变表达式的可分离结构。本轮按 softplus 正特征方案比较，未重复 signed-feature 方案。

| 模型 | 单支结构（小预算） | Q/K 两支合计有效参数 | 双倍预算 |
|---|---|---|---|
| MLP | 128→192→64，中间 SiLU | 73,856 | 147,584 |
| KAN | 128→16→64，两层 cubic B-spline+SiLU base | 73,856 | 147,584 |
| mulKAN | 128→(9加法+3二元乘法)→(32加法+32二元乘法) | 73,856 | 147,584 |

mulKAN 第一层先产生 15 个子节点，再组合为 12 个隐藏节点；第二层产生 96 个子节点，再组合为 64 个输出。两层都有乘法。每条 spline edge 含 11 个三次 B-spline 系数及 1 个 SiLU base 权重。三种结构统一无中间 bias、最终每支有 64 个 bias，所有计入的参数都参加前向计算，没有补零或闲置参数。双倍预算将 MLP/KAN 隐藏宽度加倍，mulKAN 第一层加法/乘法节点数加倍，最终 m=64 保持不变。

实现遵循 [MultKAN 原论文](https://arxiv.org/abs/2408.10205)和[官方 MultKAN 代码](https://github.com/KindXiaoming/pykan/blob/master/kan/MultKAN.py)的子节点乘积机制，是用于该实验的固定网格版本，没有引入 symbolic 分支。等参数并不等 FLOPs、吞吐或优化难度。

**数据量与“每条数据只训练一次”**

训练集为 4,096 篇文章的各 1,024 token 前缀，共 4,194,304 token，比旧 64 篇实验扩大 64 倍。前 64 篇保留旧训练文章，新增 4,032 篇；另有 128 篇验证、256 篇测试文章。新划分来自 [WikiText document-level 数据集](https://huggingface.co/datasets/EleutherAI/wikitext_document_level)的 WikiText-103 TRAIN split，是内部留出集，**不是官方 WikiText 测试集**。所有文档的文本和 token 哈希均去重；官方验证/测试及旧评估文档不进入新训练样本。

每篇取后半段 512 个 Q，与前半段 512 个 K 的固定随机排列一一配对，每个 Q、K 各使用一次，全部满足 causal 条件。因此每个 head 共 2,097,152 个唯一训练对，而非遍历每篇全部 512² 个组合。每篇只做一次梯度更新，每次运行只有 1 epoch，没有 replay、早停或测试集选 checkpoint。校准均值/方差、计算公共幅度及训练集误差会额外读取数据，但不做重复梯度训练。不同随机种子是独立重复实验，单遍约束逐次运行成立。

每一组使用同样的数据对和文档顺序，种子为 11/29/47；AdamW，lr 0.002→0.0002，weight decay=1e-4，按 head 梯度裁剪，使用最终 checkpoint。共 45 个配置运行，每次并行训练 4 个相互独立的 head 特征映射。KAN 网格固定为 8 段，没有按模型单独调学习率、初始化或网格；这些选择可能影响排序。

**更换后的损失：始终拟合未归一化 kernel**

令 $y=e^{q^\top k/\sqrt{128}-c_h}$，$\hat y=\sum_r f_r(q)g_r(k)>0$。程序通过 logsumexp 稳定计算 $\log\hat y$；不是先让特征内积拟合 logits 再在外面取 exp，后一种做法一般不再是线性 attention kernel。

1. 乘性误差 log-cosh：$L_{lc}=\log\cosh(\log\hat y-\log y)$。对 log-ratio 的梯度为 tanh，有界，避免原始平方误差被极大 kernel 值主导；代价是普通样本的相对误差改善，未必充分改善极少数大质量样本。
2. 未归一化广义 KL / I-divergence：$L_I=\hat y-y+y\log(y/\hat y)$。代码名为 `poisson`，仅借用其损失形式，不假设 kernel 是整数计数。训练省略只依赖 y 的常数，因此日志中的训练 loss 可以为负；评估报告的是恢复完整常数后的非负 divergence。

本轮**没有用原始 kernel 的 MSE 做训练**。NMSE 仅作为评估量，以保留和 L2/SVD 理论之间的联系。公共缩放 $e^{c_h}$ 不改变 log-cosh；对 I-divergence 只产生每 head 的固定权重，四个 head 的网络参数互不共享。

I-divergence 适合本问题的关键理由可以直接推导。对一行 kernel，记 $Z=\sum_jK_j$、$\hat Z=\sum_j\hat K_j$，$p=K/Z$、$\hat p=\hat K/\hat Z$，则

$$D_I(K\Vert\hat K)=Z\,D_{KL}(p\Vert\hat p)+Z\log(Z/\hat Z)-Z+\hat Z.$$

它同时惩罚行质量失配与归一化 attention 形状失配，但训练标签仍是原始 K。质量误差项 $Z\log(Z/\hat Z)-Z+\hat Z$ 非负，所以 $D_{KL}(p\Vert\hat p)\le D_I(K\Vert\hat K)/Z$；若每个 value 的范数不超过 V，则由 Pinsker 得 $\|\hat o-o\|\le V\sqrt{2D_I(K\Vert\hat K)/Z}$。这是逐行界；当前随机配对损失估计的是文档中合法矩形块的平均 entry risk，并不自动保证每一行都小。

**主要结果：4,096 篇、小参数预算**

'''+markdown_table(['训练损失','结构','测试 log-cosh↓','原始 kernel NMSE↓','I-divergence/目标质量↓','attention 输出 NMSE↓'],main_rows)+'''

表中先在每个 head 内聚合，再对固定 4 个 head 求算术平均，± 为 3 个 seed 的样本标准差，不是置信区间。前三个指标使用 256 篇测试文档的 131,072 个唯一配对样本/head；输出 NMSE 使用前 32 篇测试文档的完整合法 512×512 矩形块。原始 kernel NMSE 为 pooled SSE/真实 kernel 平方和，I-divergence 除以真实 kernel 质量；输出 NMSE 先逐文档计算再平均。不同量的数值不可直接互相比较。

log-cosh 下 MLP 更好地拟合多数样本的乘性误差；KAN/mulKAN 对原始平方风险或输出误差的表现不完全沿用这一排序。I-divergence 在三种结构上都改变了这种权衡。原始 kernel NMSE 仍然很高，不能把改进描述为已经准确替代 softmax kernel。

![损失对照](results/loss_comparison.png)

**数据是否不足、是否过拟合**

'''+markdown_table(['训练文档数','结构','训练 log-cosh','测试 log-cosh','测试−训练','输出 NMSE'],size_rows)+'''

以 log-cosh 评价时，训练和测试误差都随规模增加而下降，差距较小，未出现“大幅压低训练误差、测试误差很差”的典型整体过拟合图景。这说明不能把问题简单归因于反复训练少量数据，数据覆盖、优化步数及结构/目标偏差仍需区分。

但是，对大值更敏感的风险仍有训练—测试差距。下面是 4,096 篇、I-divergence 训练的结果：

'''+markdown_table(['结构','训练 raw NMSE','测试 raw NMSE','训练 I-div/质量','测试 I-div/质量'],tail_gap_rows)+'''

mulKAN 的训练 raw NMSE 比 MLP 低，但测试时这一优势没有保留；这提示需要关注尾部泛化。该差距可能同时反映稀有样本的拟合偏差与有限测试集的尾部波动，不能仅凭小的平均 log-cosh gap 宣布没有过拟合，也不能仅凭两个集合的加权风险之差证明其原因一定是过拟合。

必须保留的混杂因素是：训练文档增加时，optimizer steps 也从 64 增至 4,096，学习率退火随总步数变化，训练集校准统计也变化。因此本实验回答的是“更多新数据、单遍训练是否有益”，没有分离纯数据量效应与优化预算效应；单遍训练既不保证不发生过拟合，也不保证已经收敛。旧 MSE 实验的采样与训练步数不同，不能与本表作只更换损失的严格消融。

![数据规模](results/data_scaling.png)

**增大参数预算**

'''+markdown_table(['Q/K 参数','结构','测试 log-cosh↓','原始 kernel NMSE↓','输出 NMSE↓','512对特征计算 ms↓'],budget_rows)+'''

这里全部使用 log-cosh、4,096 篇、1 epoch，m 始终为 64。时间为 RTX 3090 上一次同时处理 4 个 head、每 head 512 对输入的 eager 特征网络+logsumexp 实测；不包括完整线性 attention，也不是 fused kernel 或端到端加速比较。参数匹配后，当前 spline 实现仍有显著计算开销。

**真实 kernel 的尾部与 SVD 对照**

'''+markdown_table(['head (layer,index)','最大 logit','最大0.1%样本的质量占比','最大0.1%样本的平方能量占比'],tail_rows)+'''

这里的分位数只描述本次真实 Q/K 样本，不假定高斯，也不将经验大值集中等同于严格意义的幂律分布。高平方能量集中解释了为什么 log-cosh、raw NMSE 和归一化后的输出会给出不同评价，以及为什么有限测试样本的 raw NMSE 会敏感。

'''+markdown_table(['head','signed rank-64 SVD相对下界','I-div MLP实际NMSE','I-div KAN实际NMSE','I-div mulKAN实际NMSE'],floor_rows)+'''

上述下界与实际误差使用同一组 32 个 512×512 测试块，按真实 kernel 平方能量聚合。SVD 可使用该测试块本身任意构造有符号因子，是逐块 oracle；我们的网络必须从训练数据学习、在不同文档共享函数，并具有非负因子。因此接近零的 oracle 下界不说明当前正特征模型能够达到它，也不能把实际误差与该下界之差全部归因于优化失败。它更不是全局 $P_q\otimes P_k$ 的总体最优风险证明。

对任意 m 维可分离表示，矩形预测矩阵的秩仍然至多为 m。mulKAN 的乘法发生在单侧特征网络内部，可以更直接构造坐标交互，却不会突破该秩限制。I-divergence 和 log-cosh 改变了优化目标，Schmidt 奇异值尾和只给 L2 评估的最优秩下界，不能原封不动称为这两个新损失的最小误差。要证明 mulKAN 特有优势，需要针对真实分布的主导特征函数给出相同参数或计算预算下更好的逼近率，再控制采样和优化误差；通用逼近定理本身不够。

**完整 LLM 的四个 head 替换验证**

'''+replacement_text+'''

**按 head 的结果**

'''+markdown_table(['损失','head','结构','原始 kernel NMSE','I-divergence/质量','输出 NMSE'],head_rows)+'''

**配对不确定性检查**

'''+markdown_table(['损失','结构−MLP','指标','差值','探索性95%区间'],ci_rows)+'''

负差值表示该结构误差更低。按整篇文档联合重采样，再对共享的 3 个 seed 重采样，2,000 次；head 固定。这保留文档内部相关性，但 seed 只有 3 个、测试语料单一，区间只能作探索性判断，不代表跨模型、跨领域显著性。

**验证与复现**

数据唯一性、每篇 512 个 key 的完整无重复排列、每个 run 的单 epoch/步数、跨结构顺序哈希、45 个 checkpoint 的有效参数量均已核验。Float64 小块检查验证原始 kernel 的缩放恢复、分母归一化不变性和 I-divergence 逐行分解；完整替换额外检查 causal scan 与 dense masked feature kernel 相等。数值结果记录在 `audit.json` 与 `replacement.json`。

主要文件：`extract_large.py`（真实 Q/K 提取）、`models.py`（两层网络及损失）、`train.py`（45 个单遍实验）、`validate.py`（审计）、`distribution.py`（尾部与 SVD）、`replacement.py`（完整模型替换）、`summarize.py`（报告与图）。`fits/` 保存每个模型的 checkpoint、逐 seed/head 结果及逐测试文档指标，`results/metrics.csv` 和 `results/summary.json` 可直接分析。

模型 revision：`8faed761d45a263340a0528343f099c05c9a4323`；数据 revision：`647234772b9554e208af6c826f23b99e3cac88c8`。运行环境 torch 2.13.0+cu130、transformers 5.16.1、RTX 3090。提取脚本依赖已下载的数据 parquet 和前轮 64 篇训练缓存；manifest 记录文档来源和哈希。

```bash
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python kan_attention_theory/single_pass_mulkan/models.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python kan_attention_theory/single_pass_mulkan/train.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python kan_attention_theory/single_pass_mulkan/validate.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python kan_attention_theory/single_pass_mulkan/distribution.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python kan_attention_theory/single_pass_mulkan/replacement.py
/root/miniconda3/bin/python kan_attention_theory/single_pass_mulkan/summarize.py
```

`train.py` 遇到已有结果会跳过；要从头重训使用 `--out` 指定新的目录，并在后处理时改为读取相应路径。

**课题判断**

这组数据支持继续研究“真实 Q/K 分布上的正可分离 kernel 学习及目标函数选择”，目前没有支持“mulKAN 在等预算下必然更好”的强结论。仅凭把 MLP 换成 KAN/mulKAN，顶会论证仍不足。更有价值的下一阶段是：明确正秩/总体算子与局部 SVD oracle 的差别，为未归一化 I-divergence 提供分布加权理论；在多个 LLM、更多 head、全 causal 位置和跨长度数据上验证；进行训练算力匹配及公平超参数搜索，并比较真实端到端质量—速度曲线。当前结果应作为可复现的初步验证和反证约束，不能提前宣称优于现有最强线性注意力研究。
'''
    (P/'REPORT.zh.md').write_text(report)
    print(markdown_table(['损失','模型','测试logcosh','rawNMSE','Idiv/质量','输出NMSE'],main_rows))
    print('\n'+replacement_text)
    print('\nWrote',P/'REPORT.zh.md')


if __name__=='__main__':main()
