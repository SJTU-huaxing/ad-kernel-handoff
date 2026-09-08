"""Aggregate all four curves, test the proposed gates, and render standalone figures."""
import csv
import itertools
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;P=ROOT/'results';FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
RANKS=[16,32,64,128];EPS=1e-12


def lse(x,axis=0):
    x=np.asarray(x);mx=np.max(x,axis=axis,keepdims=True)
    return np.squeeze(mx,axis=axis)+np.log(np.exp(x-mx).sum(axis=axis))


def rankdata(x):
    x=np.asarray(x);ordered=np.sort(x)
    return np.array([np.mean(np.where(ordered==v)[0]) for v in x])


def spearman(x,y):
    a=rankdata(x);b=rankdata(y)
    if np.std(a)==0 or np.std(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])


def table(columns,rows):
    return '\n'.join(['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def fmt(x):return '≤1e−12' if x<EPS else f'{x:.4g}'


def finite_pool(rows,head,n,m):
    rr=[r for r in rows if r['head']==head and r['n']==n]
    e=np.array([r['log_mean_squared_kernel'] for r in rr]);v=np.array([r['relative_floor'][str(m)] for r in rr])
    return float(np.exp(lse(e+np.log(np.maximum(v,1e-300)))-lse(e)))


def main():
    protocol=json.loads((P/'protocol.json').read_text());labels=protocol['head_labels']
    g=json.loads((P/'gaussian.json').read_text());pred=g['predictions'];valid=[h for h,p in enumerate(pred) if p['valid']]
    finite=json.loads((P/'finite_gaussian.json').read_text())['rows']
    control=json.loads((P/'controls.json').read_text())['rows']
    assert len(valid)==4
    curves={};csvrows=[];audits=dict(matrices=0,svd_lower_bound_checks=0,svd_violations=0,nmf_monotonicity_checks=0)
    for source in ['product','same_context']:
        rr=[json.loads((P/f'{source}_{rep}.json').read_text()) for rep in range(4)]
        en=np.array([r['log_energy'] for r in rr]);den=lse(en)
        for m in RANKS:
            s=np.array([r['svd'][str(m)] for r in rr])
            nm=np.array([next(x for x in r['nmf'] if x['m']==m)['error'] for r in rr])
            records={}
            for method,values in [('SVD',s),('NMF',nm)]:
                lognum=lse(en+np.log(np.maximum(values,1e-300)))
                records[method]=dict(relative=np.exp(lognum-den).tolist(),log_absolute_mse=(lognum-math.log(4)-2*math.log(512)).tolist(),
                                     per_matrix_relative=values.tolist())
            for method in ['FAVOR_plus','centered_FAVOR_plus']:
                v=np.array([[next(x for x in r['favor'] if x['m']==m and x['method']==method and x['seed']==seed)['log_nmse']
                             for seed in protocol['favor_seeds']] for r in rr])
                seedlog=lse(v+en[:,None,:],axis=0)-den[None,:]
                meanlog=lse(seedlog,axis=0)-math.log(len(seedlog))
                records[method]=dict(relative=np.exp(meanlog).tolist(),log_absolute_mse=(meanlog+den-math.log(4)-2*math.log(512)).tolist(),
                    seed_relative=np.exp(seedlog).tolist(),seed_median=np.median(np.exp(seedlog),axis=0).tolist())
            curves[(source,m)]=records
            for method,values in records.items():
                for h,label in enumerate(labels):
                    csvrows.append(dict(source=source,m=m,method=method,layer=label[0],head=label[1],
                        relative_error=values['relative'][h],log_absolute_mse=values['log_absolute_mse'][h]))
            assert np.all(nm+1e-10>=s)
            audits['svd_lower_bound_checks']+=s.size
            for r in rr:
                for b in r['favor']:
                    if b['m']!=m:continue
                    for h,logerr in enumerate(b['log_nmse']):
                        if s[rr.index(r),h]>1e-12:assert logerr+1e-9>=math.log(s[rr.index(r),h])
                        audits['svd_lower_bound_checks']+=1
                record=next(x for x in r['nmf'] if x['m']==m)
                for start in record['starts']:
                    for a,b in zip(start['history'],start['history'][1:]):
                        assert all(y<=x+1e-10 for x,y in zip(a['error'],b['error']))
                        audits['nmf_monotonicity_checks']+=16
        audits['matrices']+=len(rr)*16
    assert audits['matrices']==128
    with (ROOT/'curves.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    correlations=[]
    for m in RANKS:
        real=np.array(curves[('product',m)]['SVD']['relative']);logs=np.array(curves[('product',m)]['SVD']['log_absolute_mse'])
        expected=np.array([pred[h]['relative_floor'][str(m)] for h in valid])
        absolute=np.array([pred[h]['log_absolute_floor'][str(m)] for h in valid])
        absolute_zero=absolute-2*np.array([pred[h]['log_amplitude'] for h in valid])
        pop_rho=spearman(expected,np.maximum(real[valid],EPS))
        pvalues=[spearman(expected,np.array(real[valid])[list(p)]) for p in itertools.permutations(range(len(valid)))]
        finitepred=np.array([finite_pool(finite,label,512,m) for label in labels])
        ratio=np.maximum(real[4:]/finitepred[4:],finitepred[4:]/real[4:])
        correlations.append(dict(m=m,population_valid_heads=len(valid),population_relative_spearman=pop_rho,
            population_relative_exact_permutation_p=sum(abs(v)>=abs(pop_rho)-1e-10 for v in pvalues)/len(pvalues),
            zero_mean_population_absolute_spearman=spearman(absolute_zero,logs[valid]),
            fitted_mean_population_absolute_spearman=spearman(absolute,logs[valid]),
            finite_gaussian_16head_spearman=spearman(np.maximum(finitepred,EPS),np.maximum(real,EPS)),
            finite_gaussian_without_layer0_spearman=spearman(finitepred[4:],real[4:]),
            finite_gaussian_without_layer0_median_factor_mismatch=float(np.median(ratio)),
            finite_gaussian_without_layer0_heads_within_factor2=int(np.sum(ratio<=2)),
            finite_gaussian_relative=finitepred.tolist(),
            real_1024_relative=[finite_pool([r for r in control if r['source']=='real_product_size_check'],label,1024,m) for label in labels]))
    summaryrows=[]
    for m in RANKS:
        c=curves[('product',m)];sv=np.array(c['SVD']['relative']);nm=np.array(c['NMF']['relative']);fv=np.array(c['FAVOR_plus']['relative'])
        gap=(fv-nm)/fv
        summaryrows.append([m,fmt(np.median(sv)),fmt(np.median(nm)),fmt(np.median(fv)),f'{100*np.median(gap):.3f}%',f'{sum(gap>=.25)}/16'])
    headrows=[]
    for h,label in enumerate(labels):
        c=curves[('product',64)];hpred=fmt(pred[h]['relative_floor']['64']) if pred[h]['valid'] else '不适用'
        headrows.append([str(label),f"{pred[h]['max_coupling']:.4f}",hpred]+[fmt(c[k]['relative'][h]) for k in ['SVD','NMF','FAVOR_plus']])
    correlationrows=[[r['m'],r['population_valid_heads'],f"{r['population_relative_spearman']:.3f}",
        f"{r['zero_mean_population_absolute_spearman']:.3f}",f"{r['fitted_mean_population_absolute_spearman']:.3f}",
        f"{r['finite_gaussian_16head_spearman']:.3f}",f"{r['finite_gaussian_without_layer0_spearman']:.3f}"] for r in correlations]
    gaussian_controls=[]
    for h in valid:
        a=pred[h];label=a['head'];rs=[r for r in control if r['source']=='matched_gaussian' and r['head']==label and r['n']==512]
        # Per-matrix mean shown here, explicitly distinct from energy-pooled main curves.
        gaussian_controls.append([str(label),fmt(a['relative_floor']['64']),
            fmt(np.mean([r['relative_floor']['64'] for r in rs if r['mean']=='zero'])),
            fmt(np.mean([r['relative_floor']['64'] for r in rs if r['mean']=='fitted']))])
    # Four requested curves: invalid Gaussian heads remain marked unavailable.
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
    colors=dict(SVD='#4477aa',NMF='#228833',FAVOR_plus='#aa3377',Gaussian='#ee7733')
    for source in ['product','same_context']:
        fig,axes=plt.subplots(4,4,figsize=(13,10),layout='constrained')
        for h,(ax,label) in enumerate(zip(axes.flat,labels)):
            for method in ['SVD','NMF','FAVOR_plus']:
                y=np.array([curves[(source,m)][method]['relative'][h] for m in RANKS]);ax.plot(RANKS,np.maximum(y,EPS),'o-',label=method,color=colors[method],markersize=3)
            if pred[h]['valid']:
                ax.plot(RANKS,[pred[h]['relative_floor'][str(m)] for m in RANKS],'s--',label='Gaussian population',color=colors['Gaussian'],markersize=3)
            else:ax.text(.04,.05,'Gaussian HS condition fails',transform=ax.transAxes,fontsize=7,color=colors['Gaussian'])
            ax.set_title(f'L{label[0]} H{label[1]} | max a={pred[h]["max_coupling"]:.3f}')
            ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks(RANKS,labels=RANKS);ax.set_ylim(5e-13,1e4);ax.grid(alpha=.2)
            if h%4==0:ax.set_ylabel('Relative raw kernel squared error')
            if h>=12:ax.set_xlabel('Feature rank m')
        handles=[];names=[]
        for ax in axes.flat:
            hh,nn=ax.get_legend_handles_labels()
            for a,b in zip(hh,nn):
                if b not in names:handles.append(a);names.append(b)
        fig.legend(handles,names,loc='outside upper center',ncol=4,
                   title=f'{source}: energy-pooled over 4 matrices | FAVOR+ mean of 10 seeds | display floor 1e-12')
        for ext in ['png','pdf']:fig.savefig(FIG/f'{source}_four_curves.{ext}',dpi=170)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    corr=next(r for r in correlations if r['m']==64);real=np.maximum(curves[('product',64)]['SVD']['relative'],EPS)
    for ax,x,indices,title in [(axes[0],[pred[h]['relative_floor']['64'] for h in valid],valid,'Population Gaussian floor: 4 valid heads'),
        (axes[1],np.maximum(corr['finite_gaussian_relative'],EPS),list(range(16)),'Finite Gaussian matrix: all 16 heads')]:
        for xv,h in zip(x,indices):ax.scatter(xv,real[h],color='#4477aa' if labels[h][0] else '#999999');ax.annotate(f'{labels[h][0]}/{labels[h][1]}',(xv,real[h]),fontsize=7)
        ax.plot([EPS,1],[EPS,1],'k--',alpha=.4);ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Gaussian prediction');ax.set_ylabel('Real empirical SVD floor');ax.set_title(title);ax.grid(alpha=.2)
    for ext in ['png','pdf']:fig.savefig(FIG/f'gaussian_prediction_m64.{ext}',dpi=170)
    plt.close(fig)
    serialize=[dict(source=k[0],m=k[1],methods=v) for k,v in curves.items()]
    checks=json.loads((P/'method_checks.json').read_text())
    (ROOT/'summary.json').write_text(json.dumps(dict(curves=serialize,correlations=correlations,audits=audits,method_checks=checks,
        decision=dict(original_population_gaussian_route='NO_GO_AS_STATED',positive_feature_approximation='CONDITIONAL_GO',
                      reason='12/16 Gaussian surrogates violate HS; finite NMF has a large gap to FAVOR+, but no out-of-sample positive quadrature has been demonstrated.')),indent=2))
    report=r'''判定：**原封不动的“真实 covariance→Gaussian 总体谱→真实 head 理论极限”主线暂不 Go；“真实经验谱→正特征近似”的方法探索有条件 Go。** 本次没有训练或微调 LLM。

这不是用户列出的“Gaussian 不准且 NMF≈FAVOR，所以全部停止”的组合。本次结果是：Gaussian 总体谱对多数 head 不适用，但有限矩阵的 NMF 明显优于 FAVOR+，因此两个研究环节需要分开判断。

**先校正理论陈述**

在 κ∈L²(PQ×PK)、允许独立且有符号的两侧平方可积 feature maps 时，Schmidt 分解给出

$$\inf_{\operatorname{rank}\hat\kappa\le m}\|\kappa-\hat\kappa\|_{L^2}^2=\sum_{r>m}\sigma_r^2.$$

它是正特征方案的下界，但不能称作所有正特征方案都可达到的最小误差。共享正指数特征+非负quadrature权重又比自由的两个非负因子更受限制。

对独立零均值 Gaussian，定义 A=ΣQ^(1/2)ΣK^(1/2)/√d，奇异值为 ai，则必须有 max ai<1/2。此时

$$\mathbb E\kappa^2=\prod_i(1-4a_i^2)^{-1/2},\quad
\rho_i=\frac{2a_i}{1+\sqrt{1-4a_i^2}},\quad
\sigma_\alpha=\prod_i\sqrt{1+\rho_i^2}\,\rho_i^{\alpha_i}.$$

该谱公式正确，可以由 [Mehler 恒等式](https://dlmf.nist.gov/18.18.E28)推出；本次使用 Gauss–Hermite 求积独立验证了一维谱、非零线性项及多维模式排序。当 max ai≥1/2 时，Gaussian surrogate 的二阶矩发散，不能再报告有限总体 L2谱尾，更不能把 ai 截到0.499。真实 LLM 的有限权重、RMSNorm 和 RoPE 给出不同的有界支持情形；Gaussian 失效不表示真实 kernel 发散。

真实 Q/K 均值不为零。令 q=μQ+BQx、k=μK+BKy，b=BQᵀμK/√d、c=BKᵀμQ/√d、t=μQᵀμK/√d，l=[b;c]，M=[[I,−2A],[−2Aᵀ,I]]。在 HS 条件满足时，将 Gaussian 密度平方根吸收进核并平移 Lebesgue 坐标，得到

$$\sigma_\alpha^{\rm nonzero\ mean}
=\exp(t+l^\top M^{-1}l)\,\sigma_\alpha^{\rm zero\ mean}.$$

因此均值只改变总体谱的公共幅度，**相对谱尾**仍由 covariance 决定；但绝对误差需包含均值。有限样本矩阵却会强烈受到均值及其诱发的权重集中影响，不能把“总体相对谱与均值无关”搬到有限样本矩阵上。

**实验协议**

冻结 Qwen2.5-1.5B，复用真实投影、RoPE 后的 BF16 Q/K；第0、7、14、27层各选query head 0、3、6、9，共16个。选取覆盖不同层且不按Gaussian有效性筛选。正确处理 GQA；d=128；m=16/32/64/128。

校准用64篇训练文章，每个边缘分布32,768个向量，并检查16/32/64篇时的covariance估计。评估用20篇来源于官方WikiText-2 test split的固定1024-token文章前缀，是所选子集，不是完整benchmark。主实验每次从测试边缘池中独立抽取512个Q和512个K，构造P_Q×P_K的经验乘积矩阵，重复4次。矩阵包含跨文档配对，是分布核诊断，不是实际attention运行。另取4篇同上下文的合法512×512矩形块，作为部署相关对照；其分布不同，未混入主实验。

SVD使用Float64，每个矩阵的离散概率权重为1/512，绝对MSE是平方误差和/512²。NMF求解同一原始kernel的Frobenius近似，初值包含NNDSVD、随机正因子以及较低rank解扩展；每种初值1600次乘法更新，取每个head的最低可行误差。它是测试矩阵上的transductive因子化，不是训练LLM，也不是已可泛化的feature network。实际NMF误差是正因子最优误差的可行上界，部分试验仍有优化空间，不能称为精确非负rank下界。

FAVOR+使用 [Performer 论文](https://arxiv.org/abs/2009.14794)的正高斯正交随机特征，Haar正交方向配独立χ_d半径；每个head10个种子，所有m使用同一投影矩阵的前m行。ψω(q)=exp(d^(−1/4)ωᵀq−||q||²/(2√d))。用logsumexp计算原始feature内积，没有epsilon或未恢复的逐行缩放。另报告正确恢复所有线性项和常数项的训练均值中心化版本，作为辅助基线。数学表达式与[官方实现](https://github.com/google-research/google-research/blob/master/performer/fast_attention/tensorflow/fast_attention.py)对应；官方用于归一化attention的数值平移在此必须恢复，才能评价原始K。

全部曲线评估未归一化 exp(qᵀk/√d)。为了跨head比较，正文使用相对平方误差；每个head先按真实kernel平方能量合并4次矩阵抽样，FAVOR再对10个种子平均。绝对MSE的对数同时保存，以免指数幅度溢出。图中≤1e−12的值放在显示下限，极小数不作为精确数值证书。

**四条曲线的结果**

Gaussian公式只有4/16个head满足适用条件；16/32/64篇校准的逐head检查保存在gaussian.json。对有效head以外的12个不填造数值。

下表是主实验16个head的中位数；FAVOR列先在每个head中平均10个种子。提升率定义为(E_FAVOR−E_NMF)/E_FAVOR。

'''+table(['m','SVD误差','NMF可行误差','FAVOR+误差','NMF相对提升中位数','提升≥25%的head'],summaryrows)+r'''

这是明显的有限矩阵可改进空间；特别是存在低误差非负因子，说明在这个宽松的矩阵类中，positivity本身不必造成接近FAVOR的误差。但NMF并不约束因子来自同一指数feature族或共享quadrature节点，也不保证新Q/K上的误差。

![主实验四曲线](figures/product_four_curves.png)

m=64的逐head结果：

'''+table(['head','max ai','Gaussian总体相对误差','真实SVD','真实NMF','FAVOR+'],headrows)+r'''

第0层的经验矩阵几乎退化到很低rank，不能让这些容易的head掩盖其余head的情况。同上下文对照单独见下图及curves.csv；原始kernel的极端权重集中使跨文档乘积矩阵与同上下文矩阵的误差、以及不同重复之间都可能明显不同。

![同上下文对照](figures/same_context_four_curves.png)

**用户提出的Spearman门槛是否通过**

比较总体谱时只允许使用4个有效head。绝对风险同时给出用户零均值公式及包含实际均值幅度的Gaussian修正；后者已经不再是只用covariance预测绝对误差。相对风险使用同一个Gaussian谱形状。另有一个完全不同的量：从拟合Gaussian抽取同样大小的有限矩阵，再计算其SVD，称为有限Gaussian预测。

'''+table(['m','有效head数','总体相对谱ρ','零均值总体绝对ρ','含均值总体绝对ρ','有限Gaussian全16ρ','有限Gaussian去第0层ρ'],correlationrows)+r'''

原始总体Gaussian相对谱的相关性没有达到0.6，有效head数量也不足以支持跨8–16个head的预测主张。加入实际均值幅度后，4个有效head的绝对误差排序达到1.0，但这不再是原始零均值covariance-only公式，也没有预测准确误差量级。例如m=64时，(7,0)的绝对预测约高77倍，(7,6)约高17亿倍，(27,9)约高479倍。第0层的SVD尾值已经低于数值分辨率，其绝对尾误差与相关性还要额外谨慎。因此绝对幅度造成的高排序不能替代定量校准检验。4个head的置换检验和相关性细节保存在summary.json。

有限Gaussian全16个head的相关性达到约0.70—0.79，但这不是用户给出的闭式总体谱预测。去掉第0层四个极低误差head后，相关性仅约0.31—0.50；m=64时其余12个head的典型量级偏差约8.35倍，只有3个落在真实SVD误差的2倍范围内。不能凭全16个相关性超过0.6，就宣布“Gaussian理论下界预测成功”。

![两种不同的Gaussian预测](figures/gaussian_prediction_m64.png)

**为什么总体公式与有限SVD不能直接等同**

本次补做了真正Gaussian样本的控制实验。下面均为512×512矩阵、m=64，有限矩阵列是3次随机抽样的相对误差算术平均，故与主表的能量合并口径有所不同：

'''+table(['head','Gaussian总体谱尾','零均值Gaussian有限矩阵','含均值Gaussian有限矩阵'],gaussian_controls)+r'''

即使Gaussian假设完全成立，小矩阵的oracle也明显低估总体最优风险。有限矩阵只看到有限支持，并且可以针对该矩阵重新选择最优因子；总体谱要求一个函数对整个分布成立。权重集中及稀有事件进一步放大差异：例如(0,9)的含均值Gaussian总体log E[K²]约1381，而有限样本中观测到的尺度远低于它。这不是Mehler公式错误，而是总体与有限样本没有对齐。

控制还比较了n=256/512/1024以及真实矩阵n=1024，全部数值在results/controls.json。有限谱随n变化，不支持将512点oracle视为已经收敛的总体谱估计。对于HS无效的Gaussian，任何有限抽样矩阵依然存在，但它的有限误差绝不能冒充总体L2界。

**Go／No-Go的具体决定**

原主线暂不Go的原因有三项：大多数真实head的covariance对应一个HS失效的无界Gaussian surrogate；仅余4个有效head的总体相对谱没有预测真实矩阵的排序和量级；总体谱与有限SVD的直接对照还混入严重有限样本偏差。当前不能基于这些ρ_i给大多数head设计有理论保证的quadrature。

正特征方法探索可以有条件Go，因为已经找到远优于FAVOR+的非负矩阵因子。不过下一阶段真正需要跨过的门槛是：用训练数据确定、并在独立文档上泛化的正feature map或共享正quadrature，也能缩小这个gap。NMF目前只证明宽松矩阵类里有空间；相同ψω和非负wr的quadrature是更小的函数类，不能直接继承NMF误差。

推荐将主线改成“有界/混合真实分布→经样本规模检验的经验或总体谱→正特征可行性→跨文档泛化”，将Gaussian闭式谱保留为适用域明确的对照。对于仍在探索的Gaussian有限矩阵预测，应独立报告它是有限尺度代理模型，而不是理论下界。不要通过降低temperature、截断ai或删掉极端Q/K来隐式改变原始kernel后宣称原主线成立。

NMF≈FAVOR也不应单独作为数学意义的No-Go证据，因为数值NMF可能没优化好；本次实际没有出现这种组合。反过来，优于未经LLM适配的FAVOR+也不足以证明优于学习型正特征方法，更不足以支持顶会结论。

**验证、文件与复现**

主实验128个矩阵，4个rank；FAVOR+及均值修正版各10个种子。SVD下界、NMF非负更新的目标单调性、FAVOR原始kernel的log域计算与直接计算、均值修正恒等式，以及已知精确非负因子的固定点均检查通过。NMF冷启动不保证达到全局最优；method_checks.json特意将这一点与数值正确性分开。

`curves.csv`保存每个head/rank的四曲线及辅助baseline；`summary.json`保存门槛判定、相关性、绝对误差对数及审计。`results/`保存每个抽样矩阵的索引、SVD谱、所有NMF优化轨迹、所有FAVOR种子结果、Gaussian参数及控制实验。模型revision为`8faed761d45a263340a0528343f099c05c9a4323`，原始缓存manifest记录文档来源和哈希。

```bash
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python kan_attention_theory/gaussian_go_nogo/run.py --nmf-steps 1600 --favor-seeds 10
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python kan_attention_theory/gaussian_go_nogo/controls.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python kan_attention_theory/gaussian_go_nogo/finite_gaussian.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python kan_attention_theory/gaussian_go_nogo/check_methods.py
/root/miniconda3/bin/python kan_attention_theory/gaussian_go_nogo/summarize.py
```

运行位置为/root/autodl-tmp。主脚本可跳过已有矩阵结果；从头运行可先指定新的--out目录，并让控制及汇总脚本读取对应目录。根目录早期smoke输出仅用于流程验证，正式结果统一在results/。
'''
    report=report.replace('\\\\','\\')
    (ROOT/'REPORT.zh.md').write_text(report)
    print(table(['m','SVD','NMF','FAVOR','NMF提升','≥25%'],summaryrows))
    print(table(['m','valid','population relative','zero abs','mean abs','finite16','finite12'],correlationrows))
    print(json.dumps(audits));print('Wrote',ROOT/'REPORT.zh.md')


if __name__=='__main__':main()
