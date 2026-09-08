import json
from pathlib import Path

P=Path(__file__).resolve().parent
R=P/'results'
labels={'reduced_matched':'去Q幅度/C，同参数','reduced_plain':'纯删除，少参数','ad':'原AD','hh':'Hedgehog-exp','hh_softmax':'Hedgehog-softmax'}


def fmt(x):
    return f'{x:.6g}'


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def main():
    s=json.loads((R/'summary.json').read_text())
    k={(r['regime'],r['group'],r['split']):r for r in s['kernel']}
    prod={(r['regime'],r['group'],r['direction']):r for r in s['product']}
    ppl={(r['regime'],r['group'],r['split']):r for r in s['ppl']}
    groups=['ad','reduced_matched','reduced_plain','hh']
    text=['本轮结果不支持在原始kernel的I-divergence训练中删除查询幅度/C：同参数新版本的乘积分布原始I风险比原AD高约74%–76%，两个方向均24/24 heads更差，与Hedgehog-exp没有明确I风险优势。KL训练下删除版与原AD的attention和PPL基本持平。原AD训练后直接在推理中消去查询幅度/C，既保留原有attention结果，也获得此次删除的计算收益；无需为此改用拟合更差的I训练结构。',
    '所有新实验均冻结Qwen2.5-1.5B、两层24/336Qheads，4096训练文档、单遍、3种子。主配对参数量相同，另有少参数的纯删除控制；不是从头预训练或全层替换。',
    '**实际新kernel**\n\n$$\\hat\\kappa_-(q,k)=e^{s_K(k)}\\operatorname{softmax}([z_Q(q),0])^\\top\\operatorname{softmax}([z_K(k),0]).$$\n\nQ网络128→192→63，K网络128→192→64（63方向+1幅度），m64。没有s_Q、head标度参数、外部固定倍率或sqrt(m)。数值计算时预测和目标共同除以固定训练exp(c)，并在原始kernel评价恢复共同单位；c不是模型额外乘数。raw_log_feature返回未缩放的新kernel特征。',
    '**参数与可比性**\n\nI版本同参数73729/head：删除192个查询幅度权重和1个head标度后，增加192个Q隐藏偏置和1个方向logit偏置。KL版本同参数73728：增加192个Q隐藏偏置。新增参数均影响相对方向。纯删除两目标均73536参数。父AD/HH使用完全相同数据、单遍、3种子与每方法2个LR的既有检查点，原结果保留复用。原KL AD登记参数中192个查询幅度权重不被KL识别；原始I主比较没有这个失效分支问题。',
    '新模型沿用同种子的Q/K权重初始化、移除Q幅度输出行；新增偏置零初始化。移除外部倍率改变初始raw尺度，尤其大kernel均值head；本轮没有额外预校准初始化。因此一遍训练效果不代表充分优化后的表达上限。',
    '**乘积分布原始I训练：最直接对应原始kernel目标**']
    rows=[]
    for g in groups:
        a,b=prod['product_i',g,'ab'],prod['product_i',g,'ba']
        rows.append([labels[g],fmt(a['relative_raw_i']),fmt(b['relative_raw_i']),fmt(a['raw_nmse']),fmt(b['raw_nmse'])])
    text.append(table(['方法','A→B原始I','B→A原始I','A→B原始L2','B→A原始L2'],rows))
    text.append('I为E[d_I]/Eκ，L2为E[(κ−h)²]/Eκ²，按head先求比再汇总3种子/24heads。A/B为既有128篇留出1k文档分为不相交两组，每方向1024Q×4096K完整所选向量乘积，不是未知总体，也没有测试矩阵重拟合。')
    text.append('I训练使用d_I(x,y)=x log(x/y)−x+y，拟合对象是未归一化κ=exp(qᵀk/√128)。product_i不作逐查询质量平衡；causal_i按查询计算Σ_k d_I(κ,h)/Σ_k κ，再对查询取平均；causal_kl比较教师与预测kernel各自归一化后的因果attention方向。L2仅评估，不用于训练。三者的风险口径不能混为同一目标。')
    for regime,title in [('product_i','乘积原始I训练后的因果迁移'),('causal_i','因果balanced I训练'),('causal_kl','因果方向KL训练')]:
        text.append('**'+title+'**')
        rows=[]
        for g in groups+(['hh_softmax'] if regime=='causal_kl' else []):
            a,b=k[regime,g,'confirm_wiki'],k[regime,g,'confirm_long']
            rows.append([labels[g],fmt(a['kl']),fmt(b['kl']),fmt(a['output_nmse']),fmt(b['output_nmse']),
                         fmt(ppl[regime,g,'confirm_wiki']['ppl']),fmt(ppl[regime,g,'confirm_long']['ppl'])])
        text.append(table(['方法','1k方向KL','8k方向KL','1k输出NMSE','8k输出NMSE','1k PPL','8k PPL'],rows))
    teachers={split:json.loads((R/f'ppl_{split}_teacher.json').read_text())['ppl'] for split in ['confirm_wiki','confirm_long']}
    text.append(f'原模型PPL为{teachers["confirm_wiki"]:.6f}/{teachers["confirm_long"]:.6f}。所有PPL均完整冻结模型，仅两层局部替换；这些测试文档此前用过，不能称首次盲测。KL目标不识别完整raw标度，相关raw风险仅作诊断，不能拿它证明raw目标胜利。')
    text.append('新同参数KL版本相对原AD的两组ΔNLL文档bootstrap区间都跨0，不能宣称超越原AD；相对本轮Hedgehog-exp适配版则PPL和输出NMSE均较好。8k方向KL反而高于Hedgehog-exp，说明这里没有逐指标全面占优。HH用独立Q/K网络、无偏置、同参数扩宽到576正特征，是本研究的参数匹配适配版，不是论文默认配置或完整转换系统。')
    text.append('**I训练下的实际因果raw风险**')
    rows=[]
    for regime in ['product_i','causal_i']:
        for g in groups:
            a,b=k[regime,g,'confirm_wiki'],k[regime,g,'confirm_long']
            rows.append([regime,labels[g],fmt(a['relative_raw_i']),fmt(b['relative_raw_i']),fmt(a['raw_nmse']),fmt(b['raw_nmse'])])
    text.append(table(['训练','方法','1k原始I','8k原始I','1k原始L2','8k原始L2'],rows))
    text.append('8k原始风险的异常值必须保留：乘积I新同参模型的原始L2约984.77，原AD为0.95746；因果balanced I两种AD也都有严重长上下文幅度失配，新版本更严重。独立CPU FP64重算最坏文档的SSE，相对误差分别7.96e−15和3.36e−14，确认不是缩放还原或GPU精度错误。一个乘积I错误pair的logκ为3.187、预测log h为20.131，单pair贡献该文档平方误差约24%。这说明当前模型会对长上下文中的少数pair严重高估，并不具备稳健的8k原始kernel泛化保证。详细定位见checks/raw_cases.json。')
    text.append('**主要配对区间：新同参版本减父方法，负值有利新版本**')
    rows=[]
    for r in s['comparisons']:
        if r['a']=='reduced_matched' and r['b'] in ['ad','hh'] and r['metric']=='nll':
            rows.append([r['regime'],labels[r['b']],r['split'],fmt(r['mean_difference']),str([round(x,6) for x in r['paired_document_95_ci']])])
    text.append(table(['训练','比较对象','集合','ΔNLL/token','配对文档95%区间'],rows))
    rows=[]
    for r in s['product_comparisons']:
        if r['regime']=='product_i' and r['a']=='reduced_matched' and r['b'] in ['ad','hh']:
            rows.append([labels[r['b']],r['direction'],r['metric'],fmt(r['mean_difference']),f'{r["a_better_heads"]}/24',str([round(x,6) for x in r['paired_query_document_95_ci']])])
    text.append(table(['比较对象','方向','指标','差值','更好head数','Q文档95%区间'],rows))
    text.append('区间条件于三个已训练模型；乘积区间还固定测试K bank，只重采样Q文档。不覆盖未知PK、全部训练随机性或未观测尾部，多项探索比较未统一多重校正。')
    b=json.loads((R/'benchmark.json').read_text())
    text.append('**本轮重新计时**\n\nRTX3090、FP32、batch1、单层12heads、真实QKV、CUDA graph热缓存，统一PyTorch参考状态更新/读取，正序和逆序各一轮。原AD另有推理时直接消去query幅度/C的版本，仍使用原AD训练权重。')
    timing_labels={'ad_original':'原AD完整计算','ad_inference_cancelled':'原AD推理约掉Q幅度/C','hh_original':'Hedgehog-exp','hh_softmax':'Hedgehog-softmax','favor':'FAVOR+ m64',**labels}
    rows=[]
    for r in b['aggregates']:
        rows.append([r['regime'],timing_labels[r['variant']],r['m'],fmt(r['features_us']),fmt(r['total_us']),fmt(r['state_bytes_per_layer']/2**20)])
    text.append(table(['训练权重','实现','m','仅特征µs','特征+状态µs','状态MiB/层'],rows))
    text.append('新同参I总耗时68.672µs，比原AD完整74.560µs少7.90%，但比原AD推理约消64.160µs多7.03%；新同参KL为65.504µs，比原AD推理约消64.192µs多2.04%。新同参比HH-exp总耗时少约30%–33%，主要受m64对m576的状态成本差异影响；只计算特征，HH-exp约28.4µs更快。新同参比FAVOR+ m64的59.264µs仍慢约10.5%–15.9%。同参并不等于同m或同FLOPs，当前未融合实现的launch/归约开销也影响结果。')
    text.append('标准主导乘加：原AD特征147456 FLOPs/head/token，纯删除及原AD推理约消为147072；同参新I加偏置约147265、KL约147264。m64状态更新读取约32768，HH m576约294912。非线性、归约、输入标准化、稳定重缩放未完整计入这些主导FLOPs，计时包含实际参考实现操作。删除分支只减少很小一部分计算；相同m64的状态大小不变。没有新测全模型decode/prefill速度，PPL评价耗时不作推理benchmark。')
    c=json.loads((P/'checks'/'structure.json').read_text())
    precision=json.loads((P/'checks'/'long_precision.json').read_text())
    text.append('**机制与数值核对**\n\n在相同权重上移除Q幅度和head倍率，归一化attention不变；FP64检查最大差'+fmt(c['original_inference_cancellation_attention_max_error'])+'。纯删除与原AD在KL下共享参数的梯度也一致到FP64精度。因此KL纯删除是同一个attention函数的紧凑实现；重新训练时FP32舍入、Adam状态和学习率选择可造成小差异。补入Q方向偏置则改变了函数族。')
    text.append('原始I训练不同：原模型可直接用s_Q校准每个查询的幅度，新模型只能通过方向与K幅度共同调整。无Q幅度不等于每行质量相同：E_K h_minus=π_Q(q)^T E_K[e^{s_K(K)}π_K(K)]，仍随q改变；只是缺少独立查询缩放自由度。C能否被K网络吸收还取决于有限网络参数化，不能仅因符号删除就断言一般函数类容量减少。')
    text.append('本轮同时改变查询幅度、C、参数分配和raw初始化尺度，不能单独归因某一个因素。当前K网络无偏置，在标准化k=0处必有h_minus(q,μ_K)=1/64；已数值核验。μ_K不一定属于真实数据支持集，此恒等式只是实现约束，不是总体风险下界。未来若专门辨析C的影响，应另做保持初始raw质量可比的对照，不以本次测试结果调参。')
    text.append('18个新模型通过FP64矩阵/scan/递推核对；6个seed11配置的真实8k FP32最大相对输出误差为'+fmt(max(r['relative_max_output_error'] for r in precision['rows']))+'，无零分母。总新拟合'+str(s['new_fit_count'])+'次，选中'+str(s['selected_fit_count'])+'个，记录训练循环合计'+fmt(s['total_training_seconds']/60)+'分钟，不含数据载入和评价。')
    text.append('训练曲线按512篇文档分块，展示3种子均值和种子范围：[PNG](figures/training_curves.png)、[PDF](figures/training_curves.pdf)。I下原AD优势持续至单遍训练末尾；KL下AD与删除版曲线基本重合。单遍末尾仍有下降不构成充分优化或排除欠拟合的证据。')
    text.append('复现：proof_checks.py → train_run.py → driver.py → check_raw_cases.py → summarize_run.py → plot_training.py → write_report.py → audit_run.py。结果与父基线来源、参数预算、数据顺序、查询位置、训练尺度及文件哈希见checks/final_audit.json和results/manifest.json。研究结论应以本轮实际结果限定，不能声称总体近最优或完整Hedgehog系统复现。')
    (P/'REPORT.zh.md').write_text('\n\n'.join(text)+'\n')
    print('saved',P/'REPORT.zh.md')


if __name__=='__main__':main()
