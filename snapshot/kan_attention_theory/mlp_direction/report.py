"""Reproducible tables, paired document uncertainty, figures and Chinese report."""
import json,math,csv,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=pathlib.Path(__file__).resolve().parent
read=lambda n:json.loads((P/n).read_text())
save=lambda n,d:(P/n).write_text(json.dumps(d,indent=2,allow_nan=False))
SEEDS=[11,29,47]
LABELS={'raw':'Softplus + raw I','balanced':'Softplus + balanced I','half':'Softplus + half I','factorized_both':'Q/K amplitude-direction','exp_control':'Exp MLP + KL control','kl_control':'Softplus MLP + KL control','gauge_calibrated':'Distribution calibration','global_bal':'Global balanced calibration','global_raw':'Global raw calibration','favor_plus':'FAVOR+','centered_favor_plus':'Centered FAVOR+','sderf':'SDERF','aderf':'ADERF','teacher':'Original softmax','old_mlp':'Earlier softplus MLP','factorized':'Q amplitude-direction','value':'Value-weighted raw I'}

def kind_of(r):
 if 'metadata' in r:return r['metadata']['kind']
 if 'kind' in r:return r['kind']
 c=r.get('case',r.get('name'));return c.split('_m64_')[0]

def main():
 kernel=read('results/kernel.json')['results']+read('results/kernel_gauge.json')['results']+read('results/kernel_rf.json')['results']
 ppl=read('results/ppl.json')['results']+read('results/ppl_gauge.json')['results']
 summary=[]
 for ds in sorted({r['dataset'] for r in kernel}):
  rows=[r for r in kernel if r['dataset']==ds and ('metadata' not in r or r['metadata']['m']==64)]
  for kind in sorted({kind_of(r) for r in rows}):
   rr=[r for r in rows if kind_of(r)==kind]
   out=dict(dataset=ds,method=kind,seeds=len(rr))
   for key in ['raw_nmse','balanced','directional_kl','mass','output_nmse']:
    vals=[float(np.mean(r['summary'][key])) for r in rr]
    out[key]=float(np.mean(vals));out[key+'_seed_range']=[min(vals),max(vals)]
   summary.append(out)
 save('results/kernel_summary.json',summary)
 ps=[]
 for ds in sorted({r['dataset'] for r in ppl}):
  for kind in sorted({kind_of(r) for r in ppl if r['dataset']==ds}):
   rr=[r for r in ppl if r['dataset']==ds and kind_of(r)==kind]
   ps.append(dict(dataset=ds,method=kind,seeds=len(rr),ppl_mean=float(np.mean([r['ppl'] for r in rr])),ppl_seed_range=[min(r['ppl'] for r in rr),max(r['ppl'] for r in rr)],nll_mean=float(np.mean([r['nll'] for r in rr]))))
 save('results/ppl_summary.json',ps)
 # Paired document bootstrap conditional on the fixed three training seeds.
 # Seed means formed first; this is not a CI over future training seeds.
 contrasts=[];rng=np.random.default_rng(20260921)
 for ds in ['fresh_wiki2','swde','fresh_wiki3','swde3']:
  for a,b in [('factorized_both','raw'),('factorized_both','balanced'),('factorized_both','kl_control'),('gauge_calibrated','exp_control')]:
   aa=[r for r in ppl if r['dataset']==ds and kind_of(r)==a];bb=[r for r in ppl if r['dataset']==ds and kind_of(r)==b]
   if len(aa)!=3 or len(bb)!=3:continue
   def docs(rr):return np.mean([[d['nll'] for d in r['documents']] for r in rr],axis=0)
   delta=docs(aa)-docs(bb);boot=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(1)
   contrasts.append(dict(dataset=ds,method=a,reference=b,nll_difference=float(delta.mean()),paired_document_95ci=np.quantile(boot,[.025,.975]).tolist(),documents=len(delta),seeds=3,uncertainty_scope='Conditional on the three fixed fits; documents resampled, no claim of population or future-training-seed confidence. Multiple exploratory contrasts, unadjusted intervals.'))
 save('results/paired_nll.json',contrasts)
 products={ds:read('results/product_'+ds+'.json') for ds in ['fresh_wiki3','swde3']}
 product_summary=[]
 for ds,box in products.items():
  for r in box['results']:
   product_summary.append(dict(dataset=ds,method=r['method'],seed=r['seed'],**{k:float(np.mean(r[k])) for k in ['relative_squared_risk','relative_I_divergence','query_balanced_divergence','directional_kl','mass_divergence']}))
 save('results/product_summary.json',product_summary)
 for name,rows in [('kernel_summary',summary),('ppl_summary',ps),('product_summary',product_summary),('paired_nll',contrasts)]:
  with (P/'results'/f'{name}.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 # Two small standalone publication-style figures, with matching denominators.
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 colors={'raw':'#777777','balanced':'#bd8b46','factorized_both':'#187a75','gauge_calibrated':'#6054ac','global_bal':'#c7a0c9','global_raw':'#9a737d','favor_plus':'#ce6353','exp_control':'#397ac3','kl_control':'#659a77'}
 fig,axes=plt.subplots(1,2,figsize=(11,4.1),constrained_layout=True)
 methods=['favor_plus','raw','factorized_both','gauge_calibrated']
 for ax,ds,title in zip(axes,products,['Wikipedia: full empirical product','SWDE: full empirical product']):
  rr={r['method']:r for r in product_summary if r['dataset']==ds}
  x=np.arange(len(methods));vals=[rr[m]['relative_squared_risk'] for m in methods]
  ax.bar(x,vals,color=[colors[m] for m in methods]);ax.set_xticks(x,['FAVOR+','Raw I','Q/K split','Calibration']);ax.set_title(title);ax.set_ylabel('Relative squared raw-kernel error')
  for i,v in enumerate(vals):ax.text(i,v+.015,f'{v:.3f}',ha='center',fontsize=9)
  ax.set_ylim(0,max(vals)*1.16)
 fig.savefig(P/'figures/full_product.png',dpi=180);fig.savefig(P/'figures/full_product.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4.1),constrained_layout=True)
 methods=['raw','balanced','factorized_both','kl_control','exp_control','gauge_calibrated']
 for ax,ds,title in zip(axes,['fresh_wiki3','swde3'],['Wikipedia: 128 new documents','SWDE: 40 new source files']):
  rr={r['method']:r for r in ps if r['dataset']==ds};teacher=rr['teacher']['ppl_mean']
  for i,m in enumerate(methods):
   mean=rr[m]['ppl_mean']-teacher;lo,hi=rr[m]['ppl_seed_range'];ax.errorbar(i,mean,yerr=[[rr[m]['ppl_mean']-lo],[hi-rr[m]['ppl_mean']]],fmt='o',color=colors[m],capsize=4)
  ax.set_xticks(range(len(methods)),['Raw I','Balanced I','Q/K split','SP + KL','Exp + KL','Calibration'],rotation=22,ha='right')
  ax.set_title(title);ax.set_ylabel('PPL increase over original model');ax.set_ylim(bottom=0)
 fig.suptitle('Only 4/336 heads replaced; markers: 3-seed mean, bars: seed range',fontsize=11)
 fig.savefig(P/'figures/ppl.png',dpi=180);fig.savefig(P/'figures/ppl.pdf');plt.close(fig)
 def tab(headers,rows):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
 def get(rows,ds,kind):return next(r for r in rows if r['dataset']==ds and r['method']==kind)
 ms=['favor_plus','raw','balanced','factorized_both','kl_control','exp_control','gauge_calibrated']
 ptable=tab(['方法','Wiki 第三轮 PPL','SWDE 第三轮 PPL'],[[LABELS[m]]+[f"{get(ps,d,m)['ppl_mean']:.6f}" for d in ['fresh_wiki3','swde3']] for m in ['teacher']+ms])
 ktable=tab(['方法','Wiki 原始平方误差','Wiki 输出 NMSE','SWDE 原始平方误差','SWDE 输出 NMSE'],[[LABELS[m]]+[f"{get(summary,d,m)[k]:.6f}" for d in ['fresh_wiki3','swde3'] for k in ['raw_nmse','output_nmse']] for m in ['favor_plus','raw','balanced','factorized_both','kl_control','gauge_calibrated']])
 rftable=tab(['公式级随机特征对照（5 seeds）','Wiki raw相对平方误差','SWDE raw相对平方误差'],[[LABELS[m]]+[f"{get(summary,d,m)['raw_nmse']:.6f}" for d in ['fresh_wiki3','swde3']] for m in ['favor_plus','centered_favor_plus','sderf','aderf']])
 ftable=tab(['方法','Wiki raw L²相对误差','Wiki raw I相对误差','SWDE raw L²相对误差','SWDE raw I相对误差'],[[LABELS[m]]+[f"{get(product_summary,d,m)[k]:.6f}" for d in ['fresh_wiki3','swde3'] for k in ['relative_squared_risk','relative_I_divergence']] for m in ['favor_plus','raw','factorized_both','global_bal','global_raw','gauge_calibrated']])
 ci=tab(['数据','比较（前者−后者）','平均 ΔNLL','配对文档95%区间'],[[r['dataset'],r['method']+' − '+r['reference'],f"{r['nll_difference']:+.6f}",'['+', '.join(f'{x:+.6f}' for x in r['paired_document_95ci'])+']'] for r in contrasts if r['dataset'] in ['fresh_wiki3','swde3']])
 bench=read('results/benchmark_gauge.json')['results']
 times=tab(['方法，8192 prompt','prefill ms','decode ms/token','KV MiB','额外状态 KiB'],[[r['method'],f"{r['prefill']['median_ms']:.3f}",f"{r['decode_ms_per_token']:.3f}",f"{r['kv_cache_bytes']/2**20:.0f}",f"{r['linear_state_bytes']/1024:.0f}"] for r in bench if r['prompt_tokens']==8192])
 audit=read('checks/final_audit.json');inv=read('checks/gauge_invariance.json')
 max_inv=max(r['max_document_output_nmse_change'] for r in inv)
 max_ppl=max(abs(r['ppl_change']) for r in audit['ppl_round3_gauge_changes'])
 meta=[read('fits/'+p.name)['metadata'] for p in (P/'fits').glob('*.json')]
 training=sum(r['training_seconds'] for r in meta if r['kind'] not in ['gauge_calibrated','global_bal','global_raw'])+sum(r['calibration_seconds'] for r in meta if r['kind']=='gauge_calibrated')
 report=rf'''本轮完成了实际训练、三轮独立文档确认、完整模型的四个 head 局部替换，以及全量经验乘积分布验证。得到了两个可复现的初步结果：连续 Q/K 幅度—方向正特征在原始 kernel 拟合上优于匹配的普通 MLP；分布标度校准可以修复原始 kernel 的质量误差并保持已有 attention 输出。现有证据支持继续研究，尚不支持“新 kernel 全面优于现有方法”“接近总体 rank-m 下界”或“已有顶会水平成果”。

**最值得保留的两个构造。**

全程拟合原始目标的主候选为 factorized_both：

$$\hat\kappa(q,k)=m e^{{s_Q(q)+s_K(k)}}\,\pi_Q(q)^\top\pi_K(k),\qquad
\pi_X=\operatorname{{softmax}}([z_X,0]).$$

每侧两层 MLP 输出 m−1 个连续方向坐标与1个幅度坐标；损失是原始 I-divergence 除以真实行总质量。它没有固定分区，不是把 q/k 离散成64类。归一化 attention 的分母仍由线性状态公式计算；Q-only 幅度只是在推理时可以精确约去。MLP 中对特征坐标做 softmax 不等于把训练目标换成 attention 分数。

第二个候选使用 exp-MLP KL 控制作为方向初始化，在训练乘积分布上校准：

$$\hat\kappa(q,k)=\frac{{e^{{\ell(q)}}}}{{G(q)}}g(q,k),\quad
G(q)=\mathbb E_K g(q,K),\quad \ell(q)\approx\log\mathbb E_K e^{{q^\top K/\sqrt d}}.$$

μ=E f_K 可折叠入网络参数，固定隐藏特征上的 log-amplitude 读出采用非 MSE 的凸目标、一遍优化。没有增加推理参数或状态。对任意实际上下文，其 attention 与 g 完全相同；因此校准只能修复 raw kernel，不能凭空改善方向或PPL。此候选明确使用“KL方向预训练 + 原始kernel校准”的两阶段路线；用户要求全程原始目标的比较应使用第一个候选。

固定 g、令 Z=E_Kκ，则在积分有限条件下有精确分解：

$$R(ag)=R((Z/G)g)+\mathbb E_Q d_I(Z,aG),\qquad R(h)=\mathbb E_{{P_Q\times P_K}}d_I(\kappa,h).$$

这是固定方向等价类内的最优标度与精确 excess-risk 表达式；它不是整个正 rank-m 函数类的最优性。完整假设、推导、下界边界和现有文献归属见 [THEORY.zh.md](THEORY.zh.md)。

**实验预算与确认边界。**

冻结 Qwen2.5-1.5B，只替换 L14H0、L14H6、L27H0、L27H6，4/336个Q head。两层128→192→64的Q/K网络每head73856参数，共295424。主方案共29次单遍训练（含不同m与损失消融），3个校准读出和6个闭式标度对照；计算记录中训练/校准总耗时约{training/60:.1f}分钟，不含激活提取和评估。所有主训练目标均非MSE；KL控制明确单列。

每个主拟合使用4096训练文档、每文档64个不同Q×512K，一遍共134217728配对/head。这是相关配对数，不是独立样本量。校准额外用同一批文档中互不重叠的新64个Q，与16384-key训练bank形成4294967296次kernel计算/head；只优化原网络已有的193个读出参数/head。推理预算相同，训练算力和数据预算并不相同。校准没有重复方向训练的Q/K配对，但同一文档被两阶段访问。

初始验证128篇，既有内部测试256篇与官方WikiText2固定20篇。第一轮新增Wiki128/FDA96；第二轮新增Wiki128/SWDE96；第三轮新增Wiki128/SWDE40。Wiki/FDA每篇1024tokens，SWDE每篇512tokens。所有集合token哈希去重，SWDE额外按源文件去重；这不保证语义近重复完全不存在。第三轮在校准配置固定后测量；早期候选与追加候选的探索/确认界限见 [PROTOCOL.zh.md](PROTOCOL.zh.md)。

**同文档条件矩形：第三轮独立确认。**

下表学习方法为3种子、4heads平均，FAVOR+为5种子。raw平方误差先在每head跨文档汇总分子分母，再平均heads；输出NMSE先逐文档计算再平均。每篇1024tokens使用64Q×512K，512tokens使用32Q×256K，均为合法因果矩形。不能将这些条件矩形冒充乘积分布或完整模型输出。

{ktable}

factorized_both 与 balanced 的比较同时匹配损失、数据、参数和更新数，因此较能隔离输出参数化的作用。与 raw 的比较包含损失权重和参数化两项改变。分布校准在Wiki输出上保留exp-KL的优势，但在SWDE不胜过所有KL控制；没有一种新方案在全部指标、全部域上支配其他方案。

沿用此前固定的随机特征bank，额外确认了以下对照。它们在当前冻结模型的原始Q/K分布上误差很大，尤其受有限随机节点和大kernel值影响；这不是关于原论文完整训练系统的否定。各seed数值范围与输出误差完整保存在kernel_summary.csv。FAVOR+的正交正随机特征出处见 [Performer, ICLR2021](https://arxiv.org/abs/2009.14794)。

{rftable}

**完整经验乘积分布：不在测试矩阵上拟合。**

Wikipedia两侧各65536个新向量，所有4294967296个配对/head；SWDE两侧各10240个向量，104857600配对/head，全部FP64分块求和。各方法固定网络seed11，FAVOR+固定seed1009，未按测试选seed。下面为四个head相对风险的算术平均；raw I相对误差为 E d_I / Eκ，raw L²相对误差为 E(κ−κ̂)² / Eκ²。

{ftable}

[全量乘积分布图](figures/full_product.png)。这验证了已训练函数在新经验边缘分布上的表现，超出了单个上下文矩阵分解；它仍不是未知总体风险的置信界。不能把数十亿相关配对当成数十亿独立样本。校准与简单标度对照使用同一训练bank，区别是查询相关读出与head全局常数。未校准KL的raw标度不被识别，其很大的raw误差不作为主要胜利证据。

**完整冻结模型困惑度：只有4个head替换。**

{ptable}

学习方案表中为3种子PPL均值，FAVOR+为固定seed1009。旧测试曾测量5个FAVOR种子，不能冒充新数据上的多种子PPL。指标通过各文档token加权NLL后取exp；不同域的绝对PPL不能横向解释。样本与种子范围、全部结果保存在 [ppl_summary.csv](results/ppl_summary.csv)。[PPL图](figures/ppl.png) 显示3种子范围。

{ci}

区间是对文档配对重采样10000次，先对3个固定训练种子求均值；未覆盖未来训练种子的变异，且多项探索性比较未做多重校正。PPL改进总体很小，不能把head局部输出误差下降的百分比写成模型PPL同等比例下降。

原始kernel双精度评价中，校准前后逐文档输出NMSE最大变化为{max_inv:.3g}。初次将μ重缩放折叠到两侧FP32特征时，BF16模型中的舍入传播造成小幅PPL变化，已保留 [初次浮点部署结果](results/ppl_gauge_folded_fp32.json)。最终在线实现直接执行校准kernel的代数等价父特征、只保留一套同预算权重；重新测量后PPL最大差异为{max_ppl:.6g}。不把舍入差异作为算法质量收益。

**真实递归实现与成本。**

每head只维护 S=Σψ(k)vᵀ 和 z=Σψ(k)，状态m(d_v+1)。m64,d_v128,FP32时四个head总状态129KiB，与FAVOR+相同。前缀scan、逐token更新、显式因果矩阵在FP64核验一致。不同head的m分别分配真实状态，没有padding到最大m。

{times}

RTX3090，batch1，32个decode tokens，prefill7次、decode5次取中位数；是当前Python/Triton实现的短微基准，没有跨进程硬件方差保证。完整时长曲线见 [benchmark_gauge.json](results/benchmark_gauge.json)。当前四头混合部署仍比原模型慢，所有GQA KV组仍被剩余精确Qheads使用，因此原KV cache保留，外加129KiB状态；不能用这组数据声称全模型cache减少或推理加速。

仅按矩阵乘法计，每token每head双侧MLP特征为73728 MAC，FAVOR+为16384 MAC，约4.5倍；两者还各需约3.3万FLOP的相同状态更新/读取，忽略激活与归一化后总量比约2.75。矩阵乘法可批处理不等于已经证明GPU适配性优于FAVOR+，真实batch1额外kernel launch和GEMV仍有代价。分布标度校准的在线等价执行与父exp-MLP算子相同，校准不会额外扩大在线特征计算。

**本轮没有成立的主张与应停止的分支。**

1. 原始Gaussian covariance-only预测仍未在真实heads验证成立。此前16heads仅4个满足Gaussian L²可积条件；Schmidt尾定理在其假设内正确，不等于Gaussian模型能拟合真实Q/K。
2. 尝试用原始/重加权条件谱分配固定256维预算，原始与sqrt-mass谱均给出均匀[64,64,64,64]；row-mass谱[96,64,32,64]与验证输出配置[32,64,64,96]不同，未获得强PPL增益。容量分配本身也已有论文覆盖。
3. 正rank-m的混合信息论下界[I(Q;K)−log m]+在真实4096²训练经验乘积上、m≥16全部为0。对角toy例子验证其数学紧性，不能补救真实数据上的无效性。没有可用的m64近总体最优证书。
4. value加权原始损失有正确逐查询输出误差上界，但训练效果差于KL控制；不能把上界成立当成该损失一定更好。V重提取首训练分片相对旧Q/K有约1%的BF16差异，其他分片一致；这一失败分支也保留记录，未据此推断普遍无效。
5. 校准后的raw相对平方误差仍约数十个百分点；当前结果远不足以声称逼近原始谱极限。MI、谱、Poisson风险、PPL分别对应不同对象，不能混用。

**顶会潜力与下一阶段判据。**

我对当前完整证据的主观评价是4–5/10；作为继续投入的研究方向约6/10，不是录用概率。已有独立泛化、匹配消融、可部署正特征与精确标度误差分解，强于只展示一张训练矩阵拟合图；但主张的新颖性和实际价值还不足以支持强接收。

已有 [Hedgehog, ICLR2024](https://arxiv.org/abs/2402.04347) 的学习型特征/KL蒸馏、[LoLCATs, ICLR2025](https://arxiv.org/abs/2410.10254) 的注意力迁移、[DoF for Linear Attention, NeurIPS2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c98ef086dc70d528e1c1aa1e66893365-Abstract-Conference.html) 的分布复杂度与维度分配。行标度不变性和KL质量分解是经典事实；[Geometric Attention (2026)](https://arxiv.org/html/2601.11618v1) 也明确讨论行缩放。故“MLP+谱”“幅度方向分离”或“校准不改变attention”单独均不足以确立顶会创新性。这里只做公式/目标级匹配对照，尚未完整复现上述训练系统。

建议保留的主线是：**分布相关的正kernel近似，区分原始标度误差与真正影响attention的交互误差，并用固定预算的连续正特征同时控制两者。** 当前factorized_both是全程原始目标的主要实验支撑；分布校准提供误差可分离的构造性证据与控制实验。它们都是连续学习特征，不是FAVOR随机积分节点的简单替换，但“不是同一个公式”并不自动构成充分创新。

下一阶段应优先取得一个与正特征、非MSE风险一致且在真实heads不为零的总体下界/可预测误差，再检验其对未参与分析的heads是否有效；同时扩展到不同模型、足够多heads及长上下文/检索任务，验证在相同状态和训练成本下可重复超过KL蒸馏，并在覆盖完整GQA组的部署中证明实际cache/延迟收益。当前用户只授权4头局部替换，这些扩大部署未执行。若强对照始终不输、理论界仍不提供可用预测，应停止“逼近理论极限的新kernel”这一论文主张；可保留为kernel评价与校准研究。

复现入口为 [train.py](train.py)、[calibrate.py](calibrate.py)、[product.py](product.py)、[runtime_new.py](runtime_new.py)，推导与协议分别见上述文件。参数、数据去重、风险恒等式、状态递归、PPL加权和有限数值检查见 [final_audit.json](checks/final_audit.json)。全部负结果、原始逐文档测量和中间浮点版本均保留。
'''
 (P/'REPORT.zh.md').write_text(report)
 save('results/report_manifest.json',dict(artifacts=['REPORT.zh.md','THEORY.zh.md','PROTOCOL.zh.md','results/kernel_summary.csv','results/ppl_summary.csv','results/product_summary.csv','results/paired_nll.csv','figures/full_product.png','figures/full_product.pdf','figures/ppl.png','figures/ppl.pdf'],audit_passed=audit['passed']))
 print(json.dumps(dict(event='report',kernel_groups=len(summary),ppl_groups=len(ps),contrasts=len(contrasts),max_ppl_gauge_change=max_ppl)),flush=True)

if __name__=='__main__':main()
