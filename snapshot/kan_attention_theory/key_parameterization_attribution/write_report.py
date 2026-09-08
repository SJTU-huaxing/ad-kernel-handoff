from aggregate import *
import re

def main():
    s=read(P/'results/summary.json');out=[]
    table={(r['split'],r['group']):r for r in s['table']}
    comps={(r['split'],r['initialization'],r['metric']):r for r in s['comparisons']}
    add=out.append
    def metric(split,group,key):return table[split,group][key]
    def pct(init,key,split='confirm_long'):
        ad=metric(split,init+'_ad',key);exp=metric(split,init+'_exp',key)
        return (1-ad/exp)*100
    nll_confirm=all(comps['confirm_long',i,'nll']['conditional_document_ci95'][0]>0 for i in ['standard','matched_zero'])
    add('2026-09-08，K幅度—方向参数化归因实验。')
    if nll_confirm:
        add('**本轮支持的结论：在当前冻结Qwen、有限网络/状态及单遍KL训练预算下，AD的长上下文收益包含K幅度—方向输出参数化本身的贡献。** 控制网络深度、隐藏宽度、特征维数、有效参数、数据和初始归一化attention后，AD仍同时改善8k attention KL、输出误差与局部替换后的模型困惑度。不能把这一条件性结果提升为抽象正kernel函数类优势、所有初始化/优化器下的优势，或总体最优保证。')
    else:
        add('本轮完成了同深度、同维数、同参数及匹配初始attention的对照。结论按下面的静态误差和PPL分别判断；不能用其中一个指标替代另一个，也不能从一次有限预算实验推导总体最优性。')
    add('**究竟只改变了什么。** 双方Q网络均为128→192→63，补一个固定0后做softmax；K网络均为128→192→64。两层之间用SiLU、无偏置，均为73536有效参数/head、m=64。输入标准化相同。令输出为z_Q及u_K，比较：')
    add(r'\[\pi_Q(q)=\operatorname{softmax}([z_Q(q),0]),\qquad g_{\rm AD}(q,k)=e^{u_{K,64}(k)}\pi_Q(q)^\top\operatorname{softmax}([u_{K,1:63}(k),0]),\]')
    add(r'\[g_{\rm EXP}(q,k)=\pi_Q(q)^\top\exp(u_K(k)).\]')
    add('EXP是同深度直接指数MLP对照，不将其冒充Hedgehog原论文的完整实现。双方都没有独立查询幅度/C，最终通过线性attention分母归一化。Q网络结构及初始权重相同，双方都正常训练Q；训练后的Q权重可以不同。本实验归因的是改动K输出参数化所产生的整体训练结果，不是将训练后Q固定相同的推理消融。')
    add('随机权重对照沿用原AD初始化，AD复用既有3个检查点。匹配初始attention对照把双方K第二层置零，其K特征分别为1/64与1，因此归一化attention在任何合法因果前缀上都完全相同。两边初始参数张量也逐项相同。数值最大差9.72e−17；零初始化第一步部分梯度为零，随后所有参数均有有效梯度。这个对照排除了初始attention函数不同的解释，没有排除所有优化路径差异。')
    add('冻结Qwen2.5-1.5B，层14/27共24/336个Q heads；仅训练feature网络。训练4096篇1024-token文档，每篇64个选定Q及全部合法prefix K，共262144查询、134360517选定pairs/head，单遍反向传播。向量重复参与不同pair，不将这些pair视为独立样本。AdamW、余弦调度、梯度裁剪、文档顺序完全相同。纯因果教师attention KL损失，无raw I、MSE或输出辅助损失。')
    add('**训练与模型选择。** 每组仅用seed11在相同32篇验证文档上选择0.002/0.0005；随后固定学习率训练seed29/47。12次新拟合、9个新选定检查点，加3个复用AD检查点，共12个比较模型。方案在训练前写入[PROTOCOL.zh.md](PROTOCOL.zh.md)，没有按留出结果调参。')
    add('| 初始化 | K参数化 | 验证KL，lr=.002 | 验证KL，lr=.0005 | 选中lr |\n|---|---|---:|---:|---:|')
    for g in groups():
        rr=[r for r in s['selection'] if r['group']==g];by={r['lr']:r for r in rr}
        add(f'| {"随机权重" if g.startswith("standard") else "匹配attention"} | {g.split("_")[-1].upper()} | {by[.002]["validation_kl"]:.6f} | {by[.0005]["validation_kl"]:.6f} | {next(r["lr"] for r in rr if r["selected"])} |')
    add('训练曲线见[training_curves.png](figures/training_curves.png)，实线为3种子均值、阴影为种子范围。横轴是单遍已处理文档数，纵轴为最近512篇的平均KL；它不是重复测量固定训练子集的学习曲线。各组1k训练/验证误差接近，不能据此宣称已经达到最佳拟合或不存在欠拟合；本轮没有扩大epoch或调学习率来追求收敛极限。')
    add('**留出质量。** 下表为3种子均值，越低越好。1k为128篇，8k为24篇；这两组文档此前已经使用，不能称本轮全新盲测。KL和输出NMSE使用冻结教师缓存的64Q/文档；PPL则在完整模型上计算全部下一token，只有上述两层采用新线性attention。输出NMSE按文档/head先归一化再平均。')
    add('| 初始化 | 方法 | 1k KL | 1k PPL | 8k KL | 8k 输出NMSE | 8k PPL |\n|---|---|---:|---:|---:|---:|---:|')
    for init in ['standard','matched_zero']:
        for kind in ['ad','exp']:
            g=f'{init}_{kind}';a=table['confirm_wiki',g];b=table['confirm_long',g]
            add(f'| {"随机权重" if init=="standard" else "匹配attention"} | {kind.upper()} | {a["kl"]:.6f} | {a["ppl"]:.6f} | {b["kl"]:.6f} | {b["output_nmse"]:.6f} | {b["ppl"]:.6f} |')
    for init in ['standard','matched_zero']:
        add(f'{"随机权重" if init=="standard" else "匹配attention"}条件下，AD相对EXP的8k KL降低{pct(init,"kl"):.2f}%，输出NMSE降低{pct(init,"output_nmse"):.2f}%，PPL降低{pct(init,"ppl"):.3f}%。这是相同m、相同深度/宽度与参数数目下的差异。')
    add(f'1k并非AD全面更好：随机权重与匹配attention条件下，AD的PPL分别比EXP高{-pct("standard","ppl","confirm_wiki"):.3f}%和{-pct("matched_zero","ppl","confirm_wiki"):.3f}%。随机权重的NLL差异区间包含0；匹配attention的条件文档区间有利于EXP，但种子方向不一致。应将本轮主要收益定位为8k泛化，而非普遍质量提升。')
    add('原始教师PPL重新测得：'+ '；'.join(f'{"1k" if r["split"]=="confirm_wiki" else "8k"}={r["new"]:.9f}，与历史差{r["difference"]:.3g}' for r in s['teacher_check'])+'。因此表中方法都是有质量损失的局部替换，不是超过原softmax模型。')
    add('**配对不确定性。** 以下差值均为EXP−AD，正值有利于AD。先对3个固定种子的同一文档取平均差，再按文档配对bootstrap10000次。区间只反映这批文档上的条件不确定性，不能当作跨模型、跨训练种子的总体置信区间；没有多重比较校正。PPL以可加的NLL/token作配对检验。')
    add('| 初始化 | 8k指标 | EXP−AD | 条件文档95%区间 | 三种子差值 | AD胜文档 |\n|---|---|---:|---|---|---:|')
    for init in ['standard','matched_zero']:
        for met in ['kl','output_nmse','nll']:
            r=comps['confirm_long',init,met];lo,hi=r['conditional_document_ci95']
            add(f'| {init} | {met} | {r["exp_minus_ad"]:.6f} | [{lo:.6f}, {hi:.6f}] | '+', '.join(f'{x:.6f}' for x in r['seed_differences'])+f' | {r["ad_better_documents"]}/{r["documents"]} |')
    add('效应图见[attribution_effects.png](figures/attribution_effects.png)。所有1k配对区间、每head差异及位置分段保存在[summary.json](results/summary.json)。')
    add('**误差形态。** 下面的严重低估质量指：被模型赋予小于教师概率1/10000的keys，其教师概率质量之和，再对Q/head/doc/seed平均；不是token比例，也不是平方能量。W_O后NMSE把同层12个heads拼接并经过实际输出投影，包含head间误差交互。')
    add('| 初始化 | 方法 | 8k TV | 系数L2² | W_O后NMSE | 严重低估的教师概率质量 | KL/输出更优head数 |\n|---|---|---:|---:|---:|---:|---|')
    for init in ['standard','matched_zero']:
        for kind in ['ad','exp']:
            r=table['confirm_long',f'{init}_{kind}']
            wins='—' if kind=='exp' else f'{comps["confirm_long",init,"kl"]["ad_better_heads"]}/24；{comps["confirm_long",init,"output_nmse"]["ad_better_heads"]}/24'
            add(f'| {init} | {kind.upper()} | {r["head_tv"]:.6f} | {r["head_coeff_l2"]:.6f} | {r["layer_projected_nmse"]:.6f} | {100*r["severe_underestimate_mass"]:.3f}% | {wins} |')
    add('误差指标有不同权重，不应只挑最有利的一种。能量全局加权的输出/W_O NMSE也已完整保存；静态诊断只反映教师QKV，与替换模型实际下游激活不同。8k变化同时包含长度、内容、RoPE位置和上下文分布变化，本轮不能将它单独归因于其中一个因素。')
    add('**为何这个参数化可能起作用。** 这是可以严格证明的坐标差异，而不是已经证明的优化优势：')
    add(r'\[\log\|\phi_K^{\rm AD}(k)\|_1=s_K(k),\qquad \log\|\phi_K^{\rm EXP}(k)\|_1=\operatorname{LSE}(u_K(k)).\]')
    add('AD中，固定幅度坐标而改变方向logits，不会改变正特征的L1质量；EXP中，质量由所有输出logits共同决定。对单个合法key j，令教师/预测attention为t_j/p_j，r_j为该key内各feature对kernel的归一化贡献，则纯KL的局部导数为：')
    add(r'\[\partial_{s_j}L=p_j-t_j,\qquad\partial_{z_{jr}}L=(p_j-t_j)(r_{jr}-\pi_{K,jr}),\qquad\partial_{u_{jr}}L=(p_j-t_j)r_{jr}.\]')
    add('AD的方向式用于可训练的前63个logits；最后方向logit固定为0。输出坐标分别控制质量与方向，但共享隐藏层使实际参数更新仍会相互影响。梯度恒等式已由autograd核验到1e−16量级。结合本轮受控结果，可以提出“显式分配K质量与方向有利于当前预算下的分布外行为”的机制假设；尚未区分有限网络表达约束与优化条件各自贡献，也未证明更好的收敛率或某个具体长上下文病因。')
    add('任何严格正向量都可以写成L1幅度乘单纯形方向；AD也能重写为带共享log-sum-exp校正的指数特征。因此不能声称开辟了此前不存在的抽象正kernel函数类。另一方面，允许任意函数的代数重写，不等于一个固定宽度的线性输出层可以免费实现这个校正，本轮正是在检验有限网络预算下的这项差异。')
    add(r'更具体地，固定隐藏表示h(k)后，EXP的log质量为$\operatorname{LSE}(W_Kh)$，AD为$w_s^\top h$。若强行匹配AD方向和幅度，指数输出需要$u=[z_K,0]+[s_K-\operatorname{LSE}([z_K,0])]\mathbf1$，其中共享的非线性校正通常不能由同一个h上的线性输出层直接实现。这解释了有限网络参数化为何值得比较；允许隐藏表示改变后，本轮没有证明两类网络的严格包含关系。')
    add('**计算代价。** 两者投影主导乘加均为147072 FLOPs/head/token，m64状态读写的主导项约32768；此口径不计SiLU、exp、log-softmax、标准化和数值重标度。AD额外有K端log-softmax与幅度广播，故总计算并非完全一样。FP32单层12heads含S、z及稳定缩放g的状态均399360bytes（0.380859MiB），没有彼此的cache优势。')
    add('| 初始化 | 方法 | feature计算 μs | feature+状态步骤 μs | 两轮步骤中位数 μs |\n|---|---|---:|---:|---|')
    for r in s['benchmark']:
        add(f'| {r["group"]} | {r["group"].split("_")[-1].upper()} | {r["features_us"]:.3f} | {r["total_us"]:.3f} | '+', '.join(f'{x:.3f}' for x in r['repeat_total_us'])+' |')
    timing={r['group']:r for r in s['benchmark']}
    slow=[(timing[i+'_ad']['total_us']/timing[i+'_exp']['total_us']-1)*100 for i in ['standard','matched_zero']]
    add(f'AD的feature+状态步骤在两组中分别慢{slow[0]:.2f}%与{slow[1]:.2f}%。因此是当前实现中以约10.7%的这部分时间换取长上下文质量收益，不能宣称AD在相同网络/state预算下更快。')
    add('同一RTX3090、FP32、batch1、单层12heads、真实单token QKV、CUDA graph热缓存、同一通用状态算子，两轮正反顺序采样。时延差是当前实现结果，不能外推最优融合算子，更不是完整模型decode加速。训练壁钟保存在summary中，但历史AD与本轮不是同期运行，不把其比值当训练速度结论。')
    add('**对主线的影响。** 这个对照比“两层AD对一层加宽Hedgehog”更能支持K参数化的贡献，但仍不能把此前与Hedgehog的全部差距归给该分解，也不识别Q深度、K深度与m各自的独立效应。与已有工作的关系沿用[前轮原文核对](../normalized_kernel_novelty_20260908/REPORT.zh.md)；本轮没有新增“首创幅度方向”或“首创KL学习正feature”的主张。')
    add('现阶段可以围绕“有限特征状态下、显式K质量—方向参数化的泛化收益”继续推进。顶会层面的缺口仍包括不同模型/层及真正新域复现、收敛预算和优化器敏感性、更多同规模正feature近邻，以及归一化风险的非平凡理论。原始指数kernel的Schmidt L2谱尾不能直接作为归一化KL下界，本轮没有验证接近总体理论最优。当前是方向得到初步支持，不是已具备顶会级充分证据。')
    add('数值核验与复现实物：[结构与初始函数检查](checks/structure.json)、[梯度恒等式](checks/mechanism.json)、[FP64矩阵/scan/递推](checks/operators.json)、[真实8k FP32精度](checks/long_precision.json)、[汇总与逐种子结果](results/summary.json)、[审计与源码/结果SHA256](checks/audit.json)。新增训练日志位于logs/train.log；各检查点在fits/。训练不使用后续诊断或留出集。')
    body='\n\n'.join(out)+'\n'
    body=re.sub(r'(?m)(^\|[^\n]*\|)\n\n(?=\|)',r'\1\n',body)
    (P/'REPORT.zh.md').write_text(body)

if __name__=='__main__':main()
