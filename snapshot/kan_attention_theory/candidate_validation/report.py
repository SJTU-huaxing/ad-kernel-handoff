"""Consolidate frozen-candidate evidence, uncertainty, cost and explicit limits."""
import csv,json,math,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent;ROOT=P.parent;DEP=ROOT/'deployment_validation';COMP=ROOT/'kernel_comparison'
HEADS=[[14,0],[14,6],[27,0],[27,6]]
LABEL={'cone_mulkan':'连续非负投影 / mulKAN 基','cone_mlp':'连续非负投影 / MLP 基','cone_kan':'连续非负投影 / KAN 基',
       'cone_raw_mlp':'非负投影 / 保留幅度 MLP 基','cone_raw_kan':'非负投影 / 保留幅度 KAN 基','cone_raw_mulkan':'非负投影 / 保留幅度 mulKAN 基',
       'avg_mlp':'连续积分 / MLP 基','avg_mulkan':'连续积分 / mulKAN 基','avg_anchor':'连续积分 / 锚点基',
       'cone_anchor':'连续非负投影 / 锚点基','partition':'原分区正 kernel','galerkin':'训练 Galerkin（有符号）',
       'nystrom':'Nyström 谱截断（有符号）','spectral_pair':'谱模态成对正化','vq':'key-VQ kernel 对照',
       'teacher':'原模型','favor_plus':'FAVOR+ m64','favor_plus_640':'FAVOR+ m640',
       'nn_mlp_11':'MLP 正特征（seed 11）','nn_kan_11':'KAN 正特征（seed 11）','nn_mulkan_11':'mulKAN 正特征（seed 11）'}

def read(path):return json.loads(path.read_text())
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def f(x,d=4):return '无效' if x is None else f'{x:.{d}f}'
def bootstrap(a,b):
    diff=np.array(a)-np.array(b);gen=np.random.default_rng(20260919)
    boot=diff[gen.integers(len(diff),size=(10000,len(diff)))].mean(1)
    return dict(mean=float(diff.mean()),lower=float(np.quantile(boot,.025)),upper=float(np.quantile(boot,.975)))

def main():
    proto=read(P/'results/protocol.json');selection=read(P/'results/selection.json')
    checks=read(P/'checks/implementation.json')
    if (P/'checks/raw_implementation.json').exists():
        extra=read(P/'checks/raw_implementation.json')
        checks['checks']+=extra['checks'];checks['cone_convergence']+=extra['cone_convergence']
    kernel=[]
    for path in sorted((P/'results').glob('kernel_*.json')):kernel+=read(path)['results']
    ppl=read(P/'results/ppl.json')['results'];oldppl=read(DEP/'results/ppl_optimized.json')['results']
    merged=ppl+[r for r in oldppl if r['method'] in ['teacher','partition','galerkin','favor_plus','favor_plus_640']]
    bands=[]
    for ds in ['internal','official']:
        candidates=[r for r in ppl if r['dataset']==ds and r['method'].startswith('cone_')]
        refs=[r for r in merged if r['dataset']==ds and (r['method'].startswith('nn_mlp_') or r['method'] in ['partition','favor_plus'])]
        for c in candidates:
            for ref in refs:
                if any(x['nll'] is None for x in c['documents']+ref['documents']):continue
                bands.append(dict(dataset=ds,candidate=c['method'],reference=ref['method'],seed=ref.get('seed'),
                    **bootstrap([x['nll'] for x in c['documents']],[x['nll'] for x in ref['documents']])))
    bounds=read(DEP/'results/tightened_spectrum.json')['results']
    with (P/'results/kernel_metrics.csv').open('w') as out:
        fields=sorted(set().union(*(r.keys() for r in kernel)));writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader();writer.writerows(kernel)
    with (P/'results/ppl_metrics.csv').open('w') as out:
        fields=['method','dataset','seed','perplexity','nll','documents_count'];writer=csv.DictWriter(out,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(merged)
    (P/'results/summary.json').write_text(json.dumps(dict(paired_document_bootstrap=bands,selection=selection,
        neural_parameters_per_qk_pair=73856,raw_kernel_metrics=kernel,ppl=[{k:v for k,v in r.items() if k!='documents'} for r in merged]),indent=2))
    def kr(ds,name):
        rr=[r for r in kernel if r['dataset']==ds and r['method']==name]
        if not rr:return []
        n=max(r['n'] for r in rr);return sorted([r for r in rr if r['n']==n],key=lambda r:HEADS.index(r['head']))
    # Full internal baselines from the identical previously integrated marginals.
    internal_old={}
    g=read(ROOT/'distribution_operator/results/train_galerkin_results.json')['results']
    internal_old['galerkin']=[next(r['nmse'] for r in g if r['dataset']=='internal' and r['n']==131072 and r['m']==64 and r['head']==h) for h in HEADS]
    oldsum=read(DEP/'results/summary.json')
    internal_old['partition']=[.2065490600,.4676562613,.7842004236,.6608479866]
    for family in ['favor_plus','centered_favor_plus','sderf','aderf']:
        vals=[]
        for head in HEADS:
            rf=read(COMP/f'results/internal_L{head[0]}H{head[1]}.json')['results']
            vals.append(statistics.median(r['nmse'] for r in rf if r['method']==family and r['m']==64))
        internal_old[family]=vals
    ptable=[]
    raw_methods=['cone_raw_mlp','cone_raw_kan','cone_raw_mulkan'] if (P/'results/raw_basis_protocol.json').exists() else []
    for method in ['teacher','partition','galerkin',*selection['selected'],*raw_methods,'vq','nn_mlp','nn_kan','nn_mulkan','favor_plus','favor_plus_640']:
        vals=[]
        for ds in ['internal','official']:
            rr=[r for r in merged if r['dataset']==ds and (r['method'].startswith(method+'_') if method in ['nn_mlp','nn_kan','nn_mulkan'] else r['method']==method)]
            ps=[r['perplexity'] for r in rr]
            vals.append((f(statistics.median(ps),6)+(f' [{min(ps):.6f}, {max(ps):.6f}]' if len(ps)>1 else '')) if ps and all(x is not None for x in ps) else '无效/未完成')
        label=LABEL.get(method,{'nn_mlp':'MLP 正特征（三种子）','nn_kan':'两层 KAN 正特征（三种子）','nn_mulkan':'两层 mulKAN 正特征（三种子）'}.get(method,method))
        ptable.append([label,*vals])
    ktable=[]
    for name in ['partition','galerkin','cone_mulkan','cone_mlp','cone_kan',*raw_methods,'avg_mlp','avg_mulkan','nn_mlp_11','nn_kan_11','nn_mulkan_11','vq','nystrom','favor_plus','centered_favor_plus','sderf','aderf']:
        vals=internal_old.get(name)
        if vals is None:
            rr=kr('internal',name)
            if not rr or rr[0]['n']!=131072:continue
            vals=[r['kernel_nmse'] for r in rr]
        ktable.append([LABEL.get(name,name),*[f(x) for x in vals]])
    ot=[]
    for name in ['spectral_pair','avg_anchor','cone_anchor','cone_mulkan','cone_mlp',*raw_methods,'partition','nn_mlp_11']:
        rr=kr('official',name);ot.append([LABEL.get(name,name),*[f(r['kernel_nmse']) if r['kernel_nmse']<100 else f"{r['kernel_nmse']:.2e}" for r in rr]])
    text=[
        '本轮结论：已在冻结 Qwen2.5-1.5B 的真实 Q/K 上验证连续积分、连续非负投影、Nyström 谱截断和谱模态成对正化，并与原分区、Galerkin、FAVOR+、ADERF、key-VQ 及等参数 MLP/KAN/mulKAN 正特征比较。保留 key 特征幅度改善了部分 head 的原始 kernel 拟合，但新连续非负投影没有建立优于普通 MLP 正特征的模型困惑度与推理效率综合优势；谱正化的当前全局包络构造失败。“近总体理论下界”仍未成立。',
        '**范围与训练。** 固定模型 revision 8faed761d45a263340a0528343f099c05c9a4323；L14H0、L14H6、L27H0、L27H6，d=128，主比较 m=64。复用此前真实提取及冻结检查点，无新增 LLM 微调，无新增神经训练。MLP/KAN/mulKAN 每对 Q/K 网络均为 73,856 个活跃参数，4096 个训练文档、每 head 2,097,152 对样本，仅一遍 Poisson/广义 KL 训练，种子 11/29/47 全部报告。这些旧网络训练配对是同文档合法 Q/K；本轮 kernel 评估则为边缘乘积分布，两者不同，不将旧训练目标误称为总体乘积分布风险优化。',
        '**新构造的实际形式。** 连续基 b(k) 取 64 个校准 query 锚点的 softmax 响应，或既有 k-network 输出的连续正特征再归一化。后者各架构固定 seed 11，用于比较基函数形状。对每个 moment-fit 文档取一个 key，共 2048 个节点，h(q)=2048⁻¹Σ_i κ(q,k_i)b(k_i)。积分候选使用 c=h/p；非负投影候选对训练 Gram 的二次子问题做 128 次加速投影梯度迭代。最终 κ̂=c(q)ᵀb(k)。没有把目标改成归一化 attention；二次子问题是谱投影的数值构造，不是神经 MSE 训练。有限节点积分不是未知总体条件期望。',
        'Nyström 使用现有 1024×1024 正训练锚点 kernel 的前 64 个模态。谱正化保留 32 个模态，63 个槽位补零到 64；其包络由正锚点加权平均推导，对固定展开的所有输入成立，不用测试点估计。它不是真实总体的正交 Schmidt 函数，因此实际风险由留出数据测量，不能套用理想 E_r*+δ² 等式。',
        '**比较口径。** 新旧全部 33 个设置在验证、训练诊断、内部测试的 8192×8192 经验乘积及官方 10240×10240 经验乘积上评估。内部 8192 个样本覆盖全部 256 篇文档；官方覆盖原有全部 20 篇固定测试文档。随后将表中连续候选及神经对照复核到内部全部 131072×131072 对/每 head。FAVOR+/ADERF 五种子和原分区/Galerkin 的全量数据结果来自相同冻结数据上此前的完整积分，没有混用 8192 结果。全部 kernel 平方风险 FP64；广义 KL 与 log 误差额外使用固定的 262,144 对独立索引抽样，不能把抽样 KL 说成全矩阵精确积分。',
        '两种完整模型候选在独立的 128 篇 validation 文档上，按四个 head 的平均 raw generalized-KL/target-mass 预先选择，为 cone_mulkan 和 cone_mlp。全部测试结果均保留；没有按官方或内部测试挑选候选/种子。',
        '**原始 kernel 的全量内部经验风险。** 下表是 E[(κ−κ̂)²]/E[κ²]，所有行均对应相同的 131072×131072 经验乘积。FAVOR+/ADERF 为五种子中位数。',
        table(['方法','L14H0','L14H6','L27H0','L27H6'],ktable),
        '**官方测试的失败与泛化检查。** 下表采用全部 10240×10240 经验乘积，不是某个小块矩阵上的重新拟合。',
        table(['方法','L14H0','L14H6','L27H0','L27H6'],ot),
        '成对正化的第一模态增量系数 δ 分别为 '+', '.join(f'{x:.3e}' for x in proto['pair_delta'])+'。这证明该可部署包络构造的偏置不可接受；它不否定所有可能的谱正化或正特征方法。连续积分/非负投影的结果则说明：取消硬分区本身不足以得到更好的基函数。',
        '**完整模型困惑度，只替换 4/336 个 Q head。** 其余模型冻结，所有因果位置均重新计算，内部 256 篇与官方 20 篇各取 1024-token 前缀。官方是固定子集，非标准拼接 WikiText2 benchmark。',
        table(['方法','内部 PPL','官方子集 PPL'],ptable),
        '神经特征行为本轮三个固定种子的中位数 [最小,最大]；FAVOR+ 为五种子。单独列出的连续投影候选使用 seed 11 的基函数。所有新 PPL 测试的非正分母和非有限 attention 行数：'+str(sum(r['nonpositive_denominators'] for r in ppl))+' / '+str(sum(r['nonfinite_attention_rows'] for r in ppl))+'。总体 PPL 的小差异不能外推到全部 head 替换。',
        '**数值与表达误差。** '+str(len(checks['checks']))+' 组真实前缀的显式因果 kernel 与分块/递推实现核验，最大 FP64 相对 L2 为 '+f"{max(r['causal_recurrence_relative_l2'] for r in checks['checks']):.3e}"+'。FP32 与 FP64 的最大特征差异为 '+f"{max(r['fp32_feature_relative_l2'] for r in checks['checks']):.3e}"+'，主要来自病态非负投影；不能把投影器声称为精确 NNLS。',
        '训练 Gram 的条件数中，MLP 基可达约 2×10^8，KAN 基约 2×10^10，mulKAN 的 L14H0 在相对 10^-10 阈值下只有 40 个有效方向。对 32 个未见 query 再做 2048 次投影迭代，训练积分节点上的平方风险最多进一步降低 '+f"{100*max(max(r['relative_risk_reduction']) for r in checks['cone_convergence']):.2f}%"+'。该诊断不改变部署参数，也不证明 128 次迭代达到总体最优。',
        '报告中的 span_risk 是在留出经验测度上诊断函数空间可覆盖的能量，使用相对 10^-10 的 Gram 谱截断。coefficient_excess 是总风险减该诊断风险的数值，包含非负约束、训练积分估计与数值求解影响；由于还做了谱截断，不将它声称为全 64 维空间的精确正交分解，更不是总体正特征误差下界。',
    ]
    if raw_methods:
        text.insert(4,'**追加的基函数消融。** 一般非负锥投影不要求 Σb_s(k)=1，所以追加 cone_raw_mlp/kan/mulkan，直接使用未经分量归一化的原非负 key 特征，保留幅度。全部三种架构固定 seed 11、2048 个节点与 128 次投影，全部评估，没有据测试分数挑选其中一种。它是首轮筛选后的探索性消融；复用了同一组留出文档，不应称为全新独立盲测。两个版本的目标都一直是原始 kernel。')
    boundrows=[]
    for ds in ['internal','official']:
        for name in ['cone_mulkan','cone_mlp',*raw_methods]:
            for r in kr(ds,name):
                n=131072 if ds=='internal' else 10240
                if r['n']!=n:continue
                br=next(z for z in bounds if z['dataset']==ds and z['head']==r['head'] and z['power_iterations']==1)['bounds']['64']
                boundrows.append(dict(dataset=ds,method=name,head=r['head'],risk=r['kernel_nmse'],
                    oracle_lower=br['lower'],oracle_upper=br['upper'],ratio_at_least=r['kernel_nmse']/br['upper']))
    (P/'results/empirical_bound_comparison.json').write_text(json.dumps(boundrows,indent=2))
    if boundrows:
        text+=['**与同一经验算子的谱界比较。** 用新候选风险除以既有 rank-64 最优风险的上界，得到相对无约束最优解的确定经验差距下限。不是正特征最优值或未知总体的证书。',
            table(['方法','内部：各 head 至少多少倍','官方：各 head 至少多少倍'],[
                [LABEL.get(name,name),*[', '.join(f"{r['ratio_at_least']:.2f}×" for r in boundrows if r['dataset']==ds and r['method']==name) for ds in ['internal','official']]]
                for name in ['cone_mulkan','cone_mlp',*raw_methods]])]
    cost=[['FAVOR+ m64',32768],['MLP/KAN/mulKAN 正特征',147456],['原分区/Galerkin/Nyström m64',786432],['连续积分 / 神经基',860160],['连续非负投影 / 神经基，128 步',1908736],['key-VQ kernel',32768]]
    text+=['**计算量。** 以下为每 head、每对 Q/K 特征的主导稠密乘加 FLOPs，乘加计 2；不含 exp/softmax/spline 基函数、比较、索引、元素运算和共同线性状态聚合。KAN/mulKAN 与 MLP 的 GEMM 参数量相同，但 spline/乘法节点还有额外开销。',
        table(['方法','主导 FLOPs','相对 FAVOR+ m64'],[[a,f'{b:,}',f'{b/32768:.2f}×'] for a,b in cost]),
        '非负投影还有 2048 个真实 key 节点和节点权重等静态常数；m 相同不等于总参数、静态内存或计算量相同。三种神经架构的直接正特征对照严格保持相同网络参数量；三种神经基的积分/投影版本也使用相同基网络参数预算和节点数。FAVOR+ m640 是沿用的补充对照，仍明显低于当前 128 步投影的算术成本，不称为本轮严格同 FLOPs 对照；本轮也没有宣称候选在同 FLOPs 下获胜。',
    ]
    if (P/'results/model_benchmark.json').exists():
        bench=read(P/'results/model_benchmark.json')['results'];bt=[]
        for r in bench:
            if r['prompt_tokens']==8192:bt.append([LABEL.get(r['method'],r['method']),f(r['prefill']['median_ms'],2),f(r['decode_ms_per_token'],3),f(r['kv_cache_bytes']/2**20,1),f(r['linear_state_bytes']/1024,1)])
        text+=['**实际端到端时间。** RTX 3090、BF16 模型、FP32 特征/状态，batch=1，TF32 关闭，无并行 GPU 工作负载。沿用原协议：prefill 3 次预热 + 7 次；decode 1 次预热 + 5 次，每次固定 32 token。下表为 8192-token 提示词；计时长提示词由真实文章拼接，不构成长上下文质量验证。',
            table(['方法','Prefill ms','Decode ms/token','原 KV MiB','额外线性状态 KiB'],bt),
            '所有原 KV 仍由未替换的 GQA query heads 使用。局部替换只能增加线性状态，不能移除共享 KV；因此本轮没有全模型 cache 压缩结论。这是 HF eager 原型计时，不能称为各算法最佳算子性能。']
    if (P/'results/feature_benchmark.json').exists():
        micro=read(P/'results/feature_benchmark.json')['results'];mt=[]
        for name in ['favor_1009','favor_640','partition','galerkin','avg_mlp','cone_mlp','cone_mulkan',*raw_methods,'nn_mlp_11','nn_kan_11','nn_mulkan_11','vq']:
            vals=[]
            for n in [1,2048]:
                rr=[r for r in micro if r['method']==name and r['n']==n and r['cuda_graph']];vals.append(f(rr[0]['median_ms']*1000,2) if rr else '未完成')
            mt.append([LABEL.get(name,name),*vals])
        text+=['**特征耗时。** 同一 PyTorch 原型、双方均用 CUDA Graph，4 heads 的 Q/K 特征对，单位 μs，20 次中位数。单独展示，避免把特征时间和整模型时间混为一谈。',table(['方法','N=1 μs','N=2048 μs'],mt),
            '这里没有新写或调优融合算子。此前专门优化的 FAVOR+ N=2048 为约 67.58 μs，可作为进一步的效率参照；不同实现的数字不能当作本轮等优化程度对照。']
    if (P/'results/ppl_fp64.json').exists():
        fine=read(P/'results/ppl_fp64.json')['results'];err=[]
        for r in fine:
            base=next(x for x in ppl if x['method']==r['method'] and x['dataset']==r['dataset'])
            dd=[a['nll']-b['nll'] for a,b in zip(r['documents'],base['documents'])]
            err.append([r['method'],r['dataset'],f(sum(dd)/len(dd),7),f(max(map(abs,dd)),7)])
        text+=['**FP64 特征/状态补充检查。** 同一 BF16 模型，每个候选在每个数据集前 8 篇文档比较，表中差值为 FP64−FP32 NLL；不是重新训练。',table(['候选','数据集','平均 NLL 差','最大逐文档绝对差'],err)]
    text+=['**理论主张与 Go/No-Go。** 一般 Schmidt 误差下界仍是适用条件下的数学事实；本轮没有验证 covariance-only Gaussian 谱预测，也没有得到未知总体的近最优证书。经验跨度增大时的风险变化和官方 L14H6 的大误差，说明少量文档/尾部及训练函数空间的泛化仍是核心困难。当前结果支持停止把这版成对正化作为实用方案；也不支持把当前连续积分/128 步非负投影作为优于普通 MLP 的新方法。函数空间如何低成本覆盖有效谱，仍可作为研究问题，但需要新的构造或实质证据。',
        'FAVOR+ 与 ADERF 为既有正随机特征公式复现；key-VQ 是用校准 key k-means 形成的 kernel 级对照，没有复现端到端训练的完整 Transformer-VQ。[Performer](https://arxiv.org/abs/2009.14794)、[FAVOR#/DERF](https://proceedings.neurips.cc/paper_files/paper/2023/file/02dec8877fb7c6aa9a79f81661baca7c-Paper-Conference.pdf)、[Transformer-VQ](https://arxiv.org/abs/2309.16354)。当前理论推导和候选定义见同项目 distribution_operator/ALTERNATIVE_KERNELS.zh.md。',
        '复现脚本：prepare.py、evaluate.py、verify.py、choose_candidates.py、ppl.py、benchmark_full.py、benchmark_features.py、raw_basis.py、report.py。GPU 阶段顺序执行；原始结果、检查和文档 bootstrap 在 results/ 与 checks/。'
    ]
    (P/'REPORT.zh.md').write_text('\n\n'.join(text)+'\n')
    # Exportable figures with uncropped ordinary-risk panel and a separate
    # magnitude panel for the catastrophic positive spectral construction.
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    order=['partition','cone_mulkan','cone_raw_mlp','cone_raw_mulkan','nn_mlp_11','nn_mulkan_11','galerkin'] if raw_methods else ['partition','cone_mulkan','cone_mlp','nn_mlp_11','nn_mulkan_11','galerkin']
    for name in order:
        vals=internal_old.get(name) or [r['kernel_nmse'] for r in kr('internal',name)]
        axes[0].plot(range(4),vals,'o-',label=name)
    axes[0].set_xticks(range(4),['L14H0','L14H6','L27H0','L27H6']);axes[0].set_ylabel('Raw kernel relative squared error');axes[0].set_title('Internal full empirical product');axes[0].legend(fontsize=8)
    for name in ['teacher','partition','cone_mulkan',*(['cone_raw_mlp','cone_raw_mulkan'] if raw_methods else ['cone_mlp']),'nn_mlp_11','nn_kan_11','nn_mulkan_11','favor_plus']:
        vals=[statistics.median(r['perplexity'] for r in merged if r['dataset']==ds and r['method']==name) for ds in ['internal','official']]
        axes[1].plot(range(2),vals,'o-',label=name)
    axes[1].set_xticks([0,1],['Internal','Official subset']);axes[1].set_ylabel('Full-model PPL (4/336 heads replaced)');axes[1].legend(fontsize=8)
    fig.savefig(P/'figures/quality.png',dpi=180);fig.savefig(P/'figures/quality.pdf');plt.close(fig)
    if (P/'results/model_benchmark.json').exists():
        fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
        bench=read(P/'results/model_benchmark.json')['results']
        ix=0
        for r in bench:
            if r['prompt_tokens']!=8192:continue
            name=r['method'];pr=[v['perplexity'] for v in merged if v['dataset']=='internal' and v['method']==name]
            if not pr:continue
            x=r['decode_ms_per_token'];y=statistics.median(pr)
            marker='s' if name.startswith('nn_') else ('^' if name.startswith('cone_') else 'o')
            ax.scatter(x,y,label=name,s=65,marker=marker,color=plt.get_cmap('tab20')(ix));ix+=1
        ax.set_xlabel('Full-model decode ms/token at 8K (lower is better)');ax.set_ylabel('Internal PPL (lower is better)')
        ax.set_title('Frozen Qwen2.5-1.5B, 4/336 heads replaced; eager prototypes')
        ax.set_xlim(20,56);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3,fontsize=8)
        fig.savefig(P/'figures/quality_cost.png',dpi=180);fig.savefig(P/'figures/quality_cost.pdf');plt.close(fig)
    print('Wrote REPORT.zh.md, metrics CSVs, paired bootstrap and quality figures.')

if __name__=='__main__':main()
