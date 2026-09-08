import sys,json,pathlib,math,hashlib
import numpy as np
O=pathlib.Path(__file__).resolve().parent;C=O.parent/'causal_direction';sys.path.insert(0,str(C))
from analyze import paired_ci

def read(f):return json.loads(f.read_text())
def main():
 R=O/'results';kinds=['split_01','split_kl','softplus_kl','favor','orbit_split_01','orbit_split_kl'];kern=[];ppl=[];comparisons=[]
 for split in ['orbit_wiki','orbit_long']:
  for k in kinds:
   rr=[read(R/f'kernel_{split}_{k}_s{s}.json') for s in [11,29,47]];row=dict(split=split,kind=k)
   for key in ['kl','mass','raw_nmse','output_nmse']:
    a=np.array([r['summary'][key] for r in rr]);row[key]=float(a.mean());row[key+'_median_head']=float(np.median(a.mean(0)));row[key+'_seed_means']=a.mean(-1).tolist()
   row['hybrid_output_nmse']=float(np.mean([r['hybrid']['summary']['output_nmse'] for r in rr]));kern.append(row)
  for mode,kk in [('pure',['teacher']+kinds),('window',['swa448_sink4']),('hybrid',['split_01','softplus_kl','orbit_split_01','orbit_split_kl'])]:
   for k in kk:
    ff=[R/f'ppl_{split}_{mode}_{k}.json'] if k in ['teacher','swa448_sink4'] else [R/f'ppl_{split}_{mode}_{k}_s{s}.json' for s in [11,29,47]]
    if not all(f.exists() for f in ff):continue
    vals=[read(f)['ppl'] for f in ff];ppl.append(dict(split=split,mode=mode,kind=k,ppl_mean=float(np.mean(vals)),seed_values=vals))
  pairs=[('pure','orbit_split_01','pure','split_01'),('pure','orbit_split_kl','pure','split_kl'),('pure','orbit_split_01','pure','orbit_split_kl'),('hybrid','orbit_split_01','hybrid','split_01'),('hybrid','orbit_split_01','hybrid','softplus_kl'),('hybrid','orbit_split_01','window','swa448_sink4')]
  def docs(mode,k):
   ff=[R/f'ppl_{split}_{mode}_{k}.json']*3 if k=='swa448_sink4' else [R/f'ppl_{split}_{mode}_{k}_s{s}.json' for s in [11,29,47]]
   return None if not all(f.exists() for f in ff) else [[r['nll']/r['tokens'] for r in read(f)['documents']] for f in ff]
  for ma,ka,mb,kb in pairs:
   a=docs(ma,ka);b=docs(mb,kb)
   if a is not None and b is not None:comparisons.append(dict(split=split,a=ma+'/'+ka,b=mb+'/'+kb,**paired_ci(a,b)))
 out=dict(kernel=kern,ppl=ppl,paired_comparisons=comparisons);(R/'summary.json').write_text(json.dumps(out,indent=2,allow_nan=False))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
 for ax,split in zip(axs,['orbit_wiki','orbit_long']):
  kk=['split_01','orbit_split_01','split_kl','orbit_split_kl'];rr=[next(r for r in kern if r['kind']==k and r['split']==split) for k in kk];means=[r['output_nmse'] for r in rr];lo=[min(r['output_nmse_seed_means']) for r in rr];hi=[max(r['output_nmse_seed_means']) for r in rr];ax.bar(range(4),means,color=['#b3cde3','#2166ac','#fbb4ae','#b2182b']);ax.errorbar(range(4),means,yerr=[np.array(means)-lo,np.array(hi)-means],fmt='none',color='black',capsize=3);ax.set_xticks(range(4),['Raw + KL','Raw + KL\n+ rotations','Same-net KL','Same-net KL\n+ rotations']);ax.set_ylabel('Attention output NMSE');ax.set_title('Fresh Wiki '+('1024' if split=='orbit_wiki' else '8192')+' tokens')
 fig.savefig(O/'figures/augmentation_generalization.pdf');fig.savefig(O/'figures/augmentation_generalization.png',dpi=180);plt.close(fig)
 if len(ppl)<24:return
 def val(mode,k,split):return next(r['ppl_mean'] for r in ppl if r['mode']==mode and r['kind']==k and r['split']==split)
 tab='| 方法 | 新Wiki1024 PPL | 新长文8192 PPL |\n|---|---:|---:|\n'
 for mode,k,label in [('pure','teacher','原模型'),('window','swa448_sink4','SWA448+4'),('pure','favor','FAVOR+'),('pure','softplus_kl','softplus-KL'),('pure','split_01','原始质量+KL'),('pure','orbit_split_01','原始质量+KL+旋转增强'),('pure','split_kl','同架构KL'),('pure','orbit_split_kl','同架构KL+旋转增强'),('hybrid','split_01','原始质量+KL+窗口64'),('hybrid','orbit_split_01','原始质量+KL+增强+窗口64'),('hybrid','softplus_kl','softplus-KL+窗口64'),('hybrid','orbit_split_kl','增强KL+窗口64（未校准标度）')]:tab+=f"| {label} | {val(mode,k,'orbit_wiki'):.6f} | {val(mode,k,'orbit_long'):.6f} |\n"
 kt='| 方法 | Wiki输出NMSE | 长文输出NMSE | 长文原始kernel NMSE均值 | 长文原始NMSE按head中位数 |\n|---|---:|---:|---:|---:|\n'
 for k in kinds:
  a=next(r for r in kern if r['split']=='orbit_wiki' and r['kind']==k);b=next(r for r in kern if r['split']=='orbit_long' and r['kind']==k);kt+=f"| {k} | {a['output_nmse']:.6f} | {b['output_nmse']:.6f} | {b['raw_nmse']:.5g} | {b['raw_nmse_median_head']:.5g} |\n"
 ct='| A−B | 集合 | ΔNLL/token | 文档配对95%区间 |\n|---|---|---:|---|\n'
 for r in comparisons:ct+=f"| {r['a']} − {r['b']} | {r['split']} | {r['mean']:.6f} | [{r['paired_document_95_ci'][0]:.6f}, {r['paired_document_95_ci'][1]:.6f}] |\n"
 text=fr'''本轮在上一轮暴露共同RoPE旋转敏感性后，完成了一个新的、预先固定的增强实验：6个同预算拟合，另取64篇1024-token Wiki和16篇8192-token长文，与所有前序数据文本／token哈希不重叠。**旋转增强确实改善了长文泛化，但短文拟合变差；同样增强的KL也获得改善。** 它验证了一个可操作的缺口，未建立原始质量目标独有的优势。

**构造与预算。** 数学上的kernel类仍是连续rank64正MLP，非固定分区。训练输入改为同文档Q/K共同Rδ旋转，δ以.5概率为0，否则整数均匀0..32768；原始exp(qᵀk/√d)标签不变。每head73856参数、4096文档、262144个Q、134360517合法配对、4096更新、同种子初始化／文档顺序／归一化与AdamW设置，且两个目标的δ序列完全相同。目标λ=.1仍保留原始质量；λ=0是匹配强控制。LLM所有权重仍冻结，只替换同样两层全部24/336 heads。固定推理状态、FLOPs和算子结构与上一轮相同。

这是对“谱相同、固定feature却未泛化”的训练侧处理，尚未改变feature函数类，也没有把普通增强包装成新的解析kernel形式。完整协议见 [PROTOCOL.zh.md](PROTOCOL.zh.md)。

**新文档上的kernel确认。** 三种子、24heads均值如下；raw NMSE仅作测试诊断，训练未用MSE：

{kt}

主方法长文方向KL约3.38→1.73、输出NMSE约.639→.525，说明增强具有作用。增强KL输出NMSE约.520，说明主要改善并非质量项所独有。主方案长文原始NMSE从约1.6e16降至34.5，仍不能称原始kernel已拟合良好；均值和head中位数一起报告以暴露尾部失真。短文输出误差约.229→.266，存在明确取舍。

对本轮前16篇新Wiki作相同目标的受控旋转确认（三种子）：offset1024下原主方案方向KL为1.959，增强后.682；offset8192下为1.452→.625。相同增强的KL控制也达到.674和.606。目标logit共同旋转前后最大差为5e-14，核验了“目标与谱不变，固定feature的泛化改善”这一机制，而非目标被换掉。原始kernel不变性是数学性质；当前训练仅部分减弱敏感性，未构造严格不变的有限维正kernel。数据见 [rotation_confirmation.json](results/rotation_confirmation.json)。

**完整模型确认。** 每篇所有下一token位置计算PPL，种子均值如下。不同集合的绝对PPL不可与上一轮不同文档直接相减，下面全部使用本轮相同文档：

{tab}

增强KL混合版本未做query质量或门控校准，其原始标度具有任意性；它不能单独充当公平的强混合质量对照。上一轮已做的匹配query读出／门控KL比较没有显示原始质量独特优势；本轮混合方案的实用结论主要应与softplus-KL窗口和缓存匹配SWA比较。不能把两轮不同文档、不同父模型的校准成绩直接拼成一次胜利。

{ct}

区间为条件于三个已训练模型的配对文档bootstrap，另保留每个种子差值；新长文只有16篇，仍属初步确认。没有第二个LLM、广泛下游任务或全模型线性化的证据，也没有已证明接近未知总体谱下界。主线理论和缓存／延迟实测见 [前一轮完整报告](../causal_direction/REPORT.zh.md)。增强未增加部署算术或状态大小，但本轮没有按新权重重新做完整速度测试；不把同一算子结构表述为已经独立实测出相同延迟。

**方向评价。** 已得到可复查的机制现象及一次新数据确认：原始目标谱不变的共同旋转，会破坏现有固定特征；训练中补足该变换可改善长度泛化。适合继续研究如何以固定m保留位置变换下的结构，并检验其对原始质量与归一化输出的不同影响。单独“MLP＋谱框架＋旋转增强”的方法新颖性和当前质量／效率优势，仍不足以支持顶会方法论文。下一步的建设性kernel、相同增强／校准的强KL控制、同缓存SWA与更紧分布下界都不可省略。

当前证据宜表述为“发现并部分缓解泛化障碍”，不能表述为“已找到接近总体理论极限的新kernel”。图：[augmentation_generalization.pdf](figures/augmentation_generalization.pdf)。原始数据：[summary.json](results/summary.json)。复现依次运行collect.py select、train.py、collect.py extract、assess.py kernels、assess.py ppl、verify.py；均使用既定本地模型／数据缓存。
'''
 (O/'REPORT.zh.md').write_text(text)
 artifacts=[]
 for f in list((O/'fits').glob('*.pt'))+list((O/'results').glob('*.json'))+list(O.glob('*.py')):
  if f.name=='artifact_manifest.json':continue
  artifacts.append(dict(path=str(f.relative_to(O)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
 (R/'artifact_manifest.json').write_text(json.dumps(artifacts,indent=2));print(json.dumps(dict(kernel_rows=len(kern),ppl_rows=len(ppl),paired_comparisons=len(comparisons))))

if __name__=='__main__':main()
