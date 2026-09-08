from common_eval import *

LABELS={'ad_plain':'AD纯删除，KL训练','hh_exp':'Hedgehog-exp','hh_softmax':'Hedgehog-softmax','favor':'FAVOR+ m64'}

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])

def main():
    s=read(P/'results/summary.json')
    k={(r['split'],r['group']):r for r in s['table']}
    d={(r['split'],r['group']):r for r in s['diagnostics']}
    text=['此次复核支持AD纯删除版在已测冻结模型局部替换中的质量/状态成本优势，但不支持所有意义上的attention系数近似都优于Hedgehog。8k时AD的KL较差，TV和系数平方误差较好，输出投影后误差和PPL也较好。FAVOR+ m64质量明显较差，但参考单步耗时更低、且不需要拟合训练。',
    '对象：Qwen2.5-1.5B，层14/27的全部24/336 Qheads，4096篇1k训练文档单遍、3种子11/29/47。只使用选定的KL训练纯删除AD，参数73536/head、m64；HH-exp/softmax为本研究既有独立Q/K、同预算扩宽适配版，73728/head、m576，AD少0.26%参数。FAVOR+是固定正交高斯正特征m64，0个训练参数，不能称同训练预算或原论文完整系统比较。此次无训练、无超参选择、无全模型微调。',
    '新kernel为g(q,k)=exp(s_K(k))π_Q(q)ᵀπ_K(k)，查询方向π_Q及K方向π_K位于64维概率单纯形。没有查询幅度或整体标度。通过线性注意力分母归一化。',
    '评估集合为既有confirm_wiki128篇1k和confirm_long24篇8k；不是新的盲测。缓存系数/输出指标基于教师QKV的64个查询/文档，全模型PPL覆盖每篇所有下一token，并在两层同时替换后传播。后者层27的输入可能已受层14替换影响，不能将静态缓存指标直接当作完整模型误差分解。']
    for split,label in [('confirm_wiki','1k'),('confirm_long','8k')]:
        rows=[]
        for g in GROUPS:
            a,b=k[split,g],d[split,g]
            rows.append([LABELS[g],f'{a["kl"]:.6f}',f'{b["head_tv"]:.6f}',f'{b["head_coeff_l2"]:.6f}',f'{a["output_nmse"]:.6f}',f'{b["layer_projected_nmse"]:.6f}',f'{a["ppl"]:.6f}'])
        text.append('**'+label+'完整对照，越低越好**\n\n'+table(['方法','系数KL','系数TV','系数L2²','输出NMSE','W_O后NMSE','PPL'],rows))
    text.append('TV=0.5Σ|p−p_hat|，系数L2²=Σ(p−p_hat)²；二者和KL均先按查询平均。输出NMSE先对每个文档/head汇总SSE/教师输出能量，再平均文档、heads和种子；W_O后NMSE对每个文档/层汇总后平均，包含跨head相加。所有方法分母和选定查询一致，但不同加权口径仍可能改变排序。')
    text.append('原模型PPL为8.664560/9.066279。AD纯删除的PPL仍高于原模型，不代表已实现无损转换。8k相对HH-exp/softmax的PPL降低约0.62%，相对FAVOR降低约8.95%。')
    text.append('**为什么KL与PPL可以反向**\n\n对δ=p_hat−p，attention输出误差满足δV，平方误差为δVVᵀδᵀ；完整多head注意力子层继续做concat(δ_h V_h)W_Oᵀ，再进入残差和后续层。KL不包含V、W_O或下游模型，因此它不能确定这些误差的排序。KV数量大于value维度时，许多不同系数分布可产生相同或近似相同的输出。小KL可提供上界控制，但较大的KL不推出较大的实际输出误差。')
    text.append('一个精确示例：p=(0.5,0.25,0.25)，V=(0,1,1)。预测A=(0.5,0.49,0.01)的输出仍为0.5，尽管KL约0.6365；预测B=(0.45,0.275,0.275)的输出为0.55，KL仅约0.00503。此例仅解释数学上为何不存在单调关系，不代替真实数据诊断。')
    text.append('真实8k诊断：AD相对HH-exp在7/24heads的KL更好，但在21/24heads的输出NMSE更好；相对HH-softmax对应13/24与15/24。AD的L14H8、L14H9、L27H10三个最大KL heads解释了它相对HH-exp平均KL差距的94.08%；这是观察后的误差定位，不是用于调参的选择规则。')
    text.append('**KL的概率比惩罚已经在真实数据中定位**')
    rows=[]
    for r in s['tail']:
        t=r['thresholds'][2]
        rows.append([LABELS[r['group']],f'{100*t["teacher_mass"]:.3f}%',f'{t["positive_kl"]:.6f}',f'{r["positive_kl"]:.6f}',f'{r["negative_kl"]:.6f}'])
    text.append(table(['方法','被低估超过1万倍的教师概率质量','这些位置的正KL贡献','全部正KL贡献','全部负KL贡献'],rows))
    text.append('低估集合定义为p_hat<p/10000，概率质量为Σ集合p，再平均heads、查询、文档和种子。AD的12.23%是教师概率质量，不是key数量比例。这一真实系数失配需要改进，不能因PPL较好就将其视为无关误差。它同时允许TV/L2较低，因为KL强烈惩罚概率比，而TV/L2衡量绝对概率差。正贡献与负贡献相加才等于总KL。')
    text.append('**检查输出指标的加权敏感性**')
    rows=[]
    for g in GROUPS:
        r=d['confirm_long',g]
        rows.append([LABELS[g],f'{r["head_output_global_nmse"]:.6f}',f'{r["layer_projected_global_nmse"]:.6f}',*[f'{v:.6f}' for v in r['layer_projected_nmse_by_unit']]])
    text.append(table(['方法','全局能量加权输出NMSE','全局能量加权W_O后NMSE','L14 W_O后NMSE','L27 W_O后NMSE'],rows))
    text.append('全局能量加权采用Σ所有SSE/Σ所有教师能量。未经过W_O时，HH-softmax的0.44866优于AD的0.47171，说明此前平均head NMSE优势不等于任何加权都占优。经过真实W_O后，两种汇总下AD都较低。这支持错误方向及head/层加权会影响最终质量的解释，但还不是每个错误位置对PPL贡献的因果归因。')
    text.append('**跨种子、文档的可靠性**')
    rows=[]
    for r in s['comparisons']:
        if r['split']=='confirm_long' and r['metric'] in ['kl','nll','layer_projected_nmse']:
            rows.append([LABELS[r['b']],r['metric'],f'{r["difference"]:.6f}',str([round(x,6) for x in r['conditional_document_ci95']]),f'{r["better_documents"]}/{r["documents"]}',str([round(x,6) for x in r['seed_differences']])])
    text.append(table(['对照','指标','AD−对照','文档bootstrap95%区间','胜出文档数','各种子差值'],rows))
    text.append('文档胜出数先对3种子取均值。所有区间只重采样文档，条件于这3个已训练模型；不覆盖未知训练随机性、其他模型/head或任意语料分布，多项诊断未统一多重校正。8k有24篇文档，不能按token数夸大独立样本量。')
    text.append('**位置外推诊断**')
    rows=[]
    for g in GROUPS:
        for r in d['confirm_long',g]['positions']:
            rows.append([LABELS[g],f'{r["start"]}–{r["end"]-1}',f'{r["kl"]:.6f}',f'{r["tv"]:.6f}'])
    text.append(table(['方法','8k文档内查询位置','KL','TV'],rows))
    text.append('在同一批8k文档的前1024位置，AD的KL约0.348，仍优于两个HH；之后明显恶化。这与1k训练范围外的外推困难一致，但不能仅凭分桶判定是RoPE、内容、距离或某一结构因素单独导致。')
    bench=read(ROOT/'query_amplitude_ablation/results/benchmark.json')['aggregates']
    cases=[('ad_plain','causal_kl','reduced_plain'),('hh_exp','causal_kl','hh_original'),('hh_softmax','causal_kl','hh_softmax'),('favor','reference','favor')]
    rows=[]
    for g,regime,variant in cases:
        r=next(x for x in bench if x['regime']==regime and x['variant']==variant)
        rows.append([LABELS[g],r['m'],f'{r["features_us"]:.3f}',f'{r["total_us"]:.3f}',f'{r["state_bytes_per_layer"]/2**20:.6f}'])
    text.append('**计算成本**\n\n'+table(['方法','m','仅特征µs','特征+状态µs','状态MiB/层'],rows))
    text.append('复用上一轮统一RTX3090/FP32/batch1/单层12heads/CUDA graph参考计时：AD比HH-exp少34.3%耗时、比HH-softmax少37.7%，状态约1/9；AD比FAVOR慢8.7%，状态相同。HH的特征计算本身更快，总成本差异主要来自m576状态。不是全模型decode收益，也不是最优融合算子比较。')
    text.append('**结论的准确范围**\n\n若目标是本实验中的局部替换语言模型质量/状态成本，AD纯删除比两个HH适配版有证据支持的优势；若目标是8k系数KL，两个HH更好；若目标是TV/L2系数近似，AD更好。相对固定FAVOR+ m64，AD在已测1k/8k质量指标上更好，但需拟合且单步更慢。尚不能声称优于整个Hedgehog或FAVOR方法族、相同m下最优方法、全部heads转换、从头预训练或未知总体理论下界。')
    text.append('FAVOR本轮用相同Replacement、算子、dtype、模型revision与文档复测，6个PPL与旧值完全相同。全部12个模型的FP64缓存指标重算与父结果最大差异小于1e−9。文件来源和hash见results/manifest.json，检查见checks/final_audit.json。复现：driver.py → diagnose.py → tail_diagnose.py → aggregate.py → write_report.py → audit.py。')
    (P/'REPORT.zh.md').write_text('\n\n'.join(text)+'\n')
    print(P/'REPORT.zh.md')

if __name__=='__main__':main()
