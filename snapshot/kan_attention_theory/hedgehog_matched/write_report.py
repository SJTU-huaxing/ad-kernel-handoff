import json
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parent
read=lambda f:json.loads((P/f).read_text())
a=read('results/summary.json')
b=read('results/summary_product.json')
bench=read('results/benchmark.json')
def num(x):return f'{x:.6g}'
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join(
        '| '+' | '.join(str(v) for v in row)+' |\n' for row in rows)
labels={'ad_raw':'幅度—方向／原始损失','hh_raw':'Hedgehog-exp／原始损失',
        'ad_kl':'幅度—方向／KL','hh_kl':'Hedgehog-exp／KL','hh_softmax_kl':'Hedgehog-softmax／KL'}
def get(x,key,kind,split=None,direction=None):
    return next(r for r in x[key] if r['kind']==kind and
                (split is None or r['split']==split) and
                (direction is None or r['direction']==direction))
pt=[]
for direction in ['ab','ba']:
    for kind in ['ad_raw','hh_raw']:
        r=get(b,'product',kind,direction=direction)
        pt.append([direction,labels[kind],num(r['relative_raw_i']),num(r['raw_nmse']),
                   num(r['relative_raw_i_median_head']),num(r['raw_nmse_median_head'])])
kt=[]
for split in ['confirm_wiki','confirm_long']:
    for kind in ['ad_raw','hh_raw']:
        r=get(b,'kernel',kind,split=split)
        kt.append([split,labels[kind],num(r['relative_raw_i']),num(r['raw_nmse']),num(r['output_nmse']),
                   num(get(b,'ppl',kind,split=split)['ppl'])])
first=[]
for kind in labels:
    w=get(a,'kernel',kind,split='confirm_wiki');l=get(a,'kernel',kind,split='confirm_long')
    first.append([labels[kind],num(w['output_nmse']),num(l['output_nmse']),
                  num(get(a,'ppl',kind,split='confirm_wiki')['ppl']),
                  num(get(a,'ppl',kind,split='confirm_long')['ppl'])])
pcs=[]
for r in b['product_comparisons']:
    pcs.append([r['direction'],r['metric'],num(r['mean_difference']),
                f"{r['ad_better_heads']}/24",
                '['+', '.join(num(x) for x in r['paired_query_document_95_ci'])+']'])
nlls=[]
for stage,x in [('因果训练',a),('乘积分布训练',b)]:
    for r in x['comparisons']:
        if r['metric']=='nll':
            nlls.append([stage,r['split'],r['a']+' − '+r['b'],num(r['mean']),
                         '['+', '.join(num(v) for v in r['paired_document_95_ci'])+']'])
cost=[]
for r in bench['rows']:
    cost.append([r['name'],r['m'],num(r['features']['microseconds']),
                 num(r['features_and_state']['microseconds']),r['state_bytes_per_layer']])
prodwin=all(get(b,'product','ad_raw',direction=d)['relative_raw_i']<
            get(b,'product','hh_raw',direction=d)['relative_raw_i'] for d in ['ab','ba'])
lead=('在直接乘积分布训练后，幅度—方向在两个留出经验乘积分布上的原始I风险均低于同参Hedgehog-exp。'
      if prodwin else '直接乘积分布训练后的结果不支持幅度—方向在两个留出经验乘积分布上都取得更低原始I风险。')
reductions=[1-get(b,'product','ad_raw',direction=d)['relative_raw_i']/
            get(b,'product','hh_raw',direction=d)['relative_raw_i'] for d in ['ab','ba']]
text=fr'''本轮完成精确同参训练与真实LLM评价。**{lead}** 这仍是有限模型、有限文档与单遍训练预算下的结论；不同训练目标、长文外推、原始kernel风险和PPL必须分别判断，不能概括为在所有意义上优于Hedgehog。

**先纠正旧基线。** 此前本地Hedgehog共享Q/K权重，而论文转换代码使用独立Q、K映射。本轮已改为独立映射。旧四千参数对照不应作为充分复现Hedgehog的证据。

**参数严格相同，但状态不同。** 原始损失配对均为每head73729个活跃参数，KL配对均73728。AD两侧128→192→64，隐藏SiLU；HH两侧128→288并拼接exp(z)、exp(−z)，得到m576。统一去可学习投影偏置；原始损失配对各增加一个实际训练的head整体log标度，避免HH的固定原始幅度底限造成额外不公平。没有添加无效参数。相同训练输入标准化可以折叠入第一层仿射映射。

AD去掉原版128个输出偏置；HH为加宽、去偏置的公式适配。故这是精确同参的架构比较，不是论文默认配置复现。HH初始首128行单位阵，其余为种子固定正交块，避免加宽的零行在训练中保持完全对称；Q/K初始相同，但参数独立更新。两种架构不可能逐参数使用同一初始化。

计数补充：KL控制的73728是登记的可训练参数量，不是可识别自由度。AD的query幅度输出行有192个权重，方向KL对这一分支不识别；因此不能把KL控制也称为“损失有效自由度完全相同”。原始I主比较中该分支会影响目标，73729的主参数匹配不受此问题影响。共享MLP内部仍存在通常的参数化冗余，不主张统计可识别性定理。

双侧线性权重计算均147456主导FLOPs/head/token。ADm64，HHm576，状态是9倍；加状态更新读取后约180224与442368 FLOPs，即HH约2.45倍。相同参数不意味着相同计算或状态。

**实际执行范围。** 冻结Qwen2.5-1.5B，只替换层14/27全部24/336Qheads，未微调任何LLM权重。训练4096文档，每个主拟合单遍。每个方法相同lr搜索0.002/0.0005、seed11，验证选择后扩展seed29/47；都选0.002。首阶段20次训练，补充阶段8次，总记录网络训练时间约{(a['total_training_seconds']+b['total_training_seconds'])/60:.2f}分钟，不含提取、标签预积分和评价。两个阶段的协议在各自训练前固定，补充阶段是看到首阶段乘积失败后的明确追加实验。

**为什么补直接乘积分布训练。** 首阶段沿用同文档因果配对，AD/HH用同balanced I（λ=1）比较raw目标，另有同参KL及HH-softmax方向控制。首阶段在1k因果评价有优势，但跨文档经验乘积的AD风险异常大。独立CPU FP64复算最差head L27H10确认不是实现错误：某pair目标logκ约−2.36、预测约34.72。该例用于事后故障诊断，不是选择测试样本训练。原始结果完整保留。

只在同文档因果测度上学好，不能推出PQ×PK风险低。补充阶段对每个训练Q文档选16个其他文档、每篇64个K，形成1024-key bank；4096文档全部用作Q/K来源，无同文档pair，每个Q及选定pair只反向传播一次。训练pair为268435456/head，向量复用不代表同数量独立样本。

补充阶段使用**未经query平衡的原始I-divergence**，只除以固定head训练kernel均值作数值缩放。该均值仅从固定训练pairs预积分得到，之后恢复原始幅度评价；它不是当前query或attention行归一化。输入目标始终exp(qᵀk/√128)。损失与配对均相对首阶段改变，因此两阶段之差不能只归因于一个因素。细节见[PRODUCT_PROTOCOL.zh.md](PRODUCT_PROTOCOL.zh.md)。

**与你主线最相关：直接训练后的完整所选经验乘积。**

128个留出1k文档均分A/B各64篇；Q每文档取16个缓存位置、K每文档64个位置，1024×4096全部pair/head，另做反向B→A。模型测试前冻结，不在测试矩阵拟合，不求测试NMF。这里覆盖全部所选向量的乘积，不是所有缓存位置或未知真实总体。所有head/seed均值与head中位数一起报告。

{table(['方向','方法','原始I相对风险','原始L2相对误差','I的head中位数','L2的head中位数'],pt)}

原始I风险相对下降分别为{reductions[0]*100:.2f}%与{reductions[1]*100:.2f}%。按三个种子的每head均值，两方向均24/24heads更低；这不是把24个head当作24个独立模型做统计推断。原始L2在A→B的差异区间跨零，因此不能把I风险的稳定优势推广为两个方向都已确认L2优势。

差值为AD−HH，负值有利AD：

{table(['方向','指标','平均差','AD更好head数','配对Q文档95%区间'],pcs)}

这些区间条件于固定测试K bank与三个拟合模型，只重采样Q文档；不覆盖未知PK、完整训练随机性或未观测尾部。它们不是总体误差下界或近最优证书。

**直接乘积分布训练后的因果迁移。**

{table(['集合','方法','原始I相对风险','原始L2相对误差','输出NMSE','全模型PPL'],kt)}

这张表是同文档实际因果配对及完整模型输出，不能与上一张乘积表混为同一个风险。高原始误差不能用PPL或归一化输出改善掩盖；反之，低原始乘积风险也不自动保证每个实际上下文的方向、输出或PPL。

**首阶段同损失、同参数的因果训练对照。**

{table(['方法','1k输出NMSE','8k输出NMSE','1k PPL','8k PPL'],first)}

原模型PPL为8.664560/9.066279。首阶段AD与HH在相同KL目标下的比较检验的是两种特征参数化；raw训练对KL控制的比较还包含目标差异。KL控制未经query幅度校准的巨大raw误差不作为原始kernel胜利证据。HH-softmax具体为concat(softmax(z),softmax(−z))，是论文所讨论稳定化思路的一个明确实现，不能冒充全部官方实现/训练流程。

{table(['训练阶段','集合','比较','ΔNLL/token','配对文档95%区间'],nlls)}

区间条件于三个冻结种子，文档作为重采样单位；探索性多比较未做统一多重校正。现有confirm_wiki128篇、confirm_long24篇此前已用于研究，不能称首次盲测。没有第二模型或全层线性化确认。

**时间与状态。** RTX3090、FP32、batch1、单层12heads，真实Q/K/V，CUDA graph热缓存，同一PyTorch参考特征及状态实现：

{table(['方法','m','仅特征 μs','特征+状态 μs','单层状态 bytes'],cost)}

这些是统一参考实现的微测，不是融合到最优的算子，也不是全模型decode时间。不能与旧Triton融合m64的约5–10μs结果直接拼接比较。首阶段计时后的乘积训练改变了权重、没有改变算子结构；没有为补充权重另作计时，成本只作同结构参考。

**数值验证。** 两阶段均通过FP64显式矩阵/分块scan/递推一致性，以及实际8k keys与真实64个Q的FP32递推对FP64显式参考检查。首阶段最大相对输出误差约3.5e−6，未出现零分母；补充阶段精确值在checks/long_precision_product.json。PPL使用完整冻结模型、逐文档所有下一token位置，未混用窗口、旋转增强或LoRA。

**理论解释及边界。** 仿射+正指数HH是(q,k)的联合凸函数，而原始exp(qᵀk/√d)一般不联合凸。THEORY.zh.md给出一个真实乘积分布上的三点支持反例：即使增加HH宽度，仍有正的逼近误差，而自由非负rank3可精确表示。两层非线性AD不受同样的联合凸性约束。但该反例不是当前LLM总体的误差下界，也不适用于带特征softmax或非线性query校准的全部HH变体。这涉及非线性深度与输出结构，不是幅度—方向独占优势。

因此，这轮能评价精确预算下的实测质量、训练测度与泛化缺口；不能证明AD在所有分布、充分优化预算或全模型系统上必然优于Hedgehog，也没有接近总体谱下界的证据。

**Hedgehog与FAVOR+究竟怎样训练。** Hedgehog原论文两种都做：从头训练时，feature MLP和原模型权重一起用任务损失学习；转换已有模型时，先冻结Transformer，以相同Q/K的softmax attention为教师，训练feature map匹配归一化权重，再进行任务微调或LoRA。见[原论文A.3](https://arxiv.org/html/2402.04347v1#A3)。

FAVOR+本身是正交正随机特征近似，基础随机投影通常不是通过教师拟合SGD学出来的。Performer可以从头训练，此时主要学习原模型投影/FFN等权重，随机特征可以重采样；也可替换已有Transformer再评价或继续训练。理论可替换不保证任意有限m、任意已训练LLM立即无损。见[Performer原论文](https://arxiv.org/html/2009.14794v4)。

本轮属于冻结模型后的特征拟合与局部替换，只对应转换流程中的特征阶段。不能把Hedgehog经过后续任务微调的论文结果，与本轮无模型微调的分数直接比较。

**复现与证据。** 首阶段依次train.py、driver.py、summarize.py；补充阶段train_product.py后，设置MATCHED_PLAN=product_plan.json运行driver.py与summarize.py。最后运行write_report.py、audit.py。results/summary.json与summary_product.json保留三种子、逐head聚合与配对区间；fits/保留全部28次拟合及曲线，checks/final_audit.json和results/artifact_manifest.json记录预算与来源检查。
'''
(P/'REPORT.zh.md').write_text(text)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
colors={'ad_raw':'#2166ac','hh_raw':'#d6604d'}
for ax,metric,title in [(axs[0,0],'relative_raw_i','Raw I risk: heldout empirical products'),
                         (axs[0,1],'raw_nmse','Raw L2 risk: heldout empirical products')]:
    for j,kind in enumerate(['ad_raw','hh_raw']):
        vals=[get(b,'product',kind,direction=d)[metric] for d in ['ab','ba']]
        ax.bar(np.arange(2)+(j-.5)*.3,vals,width=.3,label='AD m64' if j==0 else 'HH m576',color=colors[kind])
    ax.set_xticks([0,1],['A to B','B to A']);ax.set_title(title);ax.legend()
for ax,key,metric,title in [(axs[1,0],'kernel','output_nmse','Causal attention output NMSE'),
                            (axs[1,1],'ppl','ppl','Frozen model PPL: 24/336 heads replaced')]:
    for j,kind in enumerate(['ad_raw','hh_raw']):
        vals=[get(b,key,kind,split=s)[metric] for s in ['confirm_wiki','confirm_long']]
        ax.bar(np.arange(2)+(j-.5)*.3,vals,width=.3,color=colors[kind])
    ax.set_xticks([0,1],['1k','8k']);ax.set_title(title)
fig.suptitle('Direct product training; exact 73,729 trainable parameters per head')
fig.savefig(P/'figures/product_comparison.png',dpi=170)
fig.savefig(P/'figures/product_comparison.pdf')
plt.close(fig)
print(P/'REPORT.zh.md')
