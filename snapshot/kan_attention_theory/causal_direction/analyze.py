"""CPU reporting and paired document statistics; never changes fitted models."""
import json,math,pathlib
import numpy as np
P=pathlib.Path(__file__).resolve().parent;R=P/'results'
def read(f):return json.loads(f.read_text())
def save(f,x):f.write_text(json.dumps(x,indent=2,allow_nan=False))
def paired_ci(a,b,seed=9531):
 # Report paired document uncertainty conditional on the three frozen fits;
 # seed variation is retained separately, not inflated into independent heads.
 d=np.asarray(a)-np.asarray(b);avg=d.mean(0);gen=np.random.default_rng(seed);idx=gen.integers(0,len(avg),size=(10000,len(avg)));boot=avg[idx].mean(-1)
 return dict(mean=float(d.mean()),paired_document_95_ci=np.quantile(boot,[.025,.975]).tolist(),per_seed_mean=d.mean(-1).tolist(),interval_scope='Document bootstrap conditional on these three fits; only3 training seeds, not a full training-randomness population CI.')

def main():
 kernel=[];ppl=[];comparisons=[]
 for split in ['confirm_wiki','confirm_long','confirm_prose']:
  for kind in ['split_1','split_01','split_kl','exp_kl','softplus_kl','calibrated','gate','global_bal','global_raw','favor','hedgehog','learned_prf']:
   files=[R/f'kernel_{split}_{kind}_s{s}.json' for s in [11,29,47]]
   if not all(f.exists() for f in files):continue
   a=[read(f) for f in files];row=dict(split=split,kind=kind)
   for key in ['kl','mass','balanced','output_nmse','raw_nmse']:
    vals=np.array([r['summary'][key] for r in a]);row[key]=float(vals.mean());row[key+'_seed_means']=vals.mean(-1).tolist();row[key+'_median_head']=float(np.median(vals.mean(0)))
   if 'hybrid'in a[0]:
    for key in ['kl','output_nmse','mixture_mass_kl']:row['hybrid_'+key]=float(np.mean([r['hybrid']['summary'][key] for r in a]))
   kernel.append(row)
  for mode,kinds in [('pure',['teacher','split_01','split_kl','exp_kl','softplus_kl','favor','hedgehog','learned_prf']),('hybrid',['split_01','softplus_kl','global_bal','global_raw','calibrated','gate','favor']),('window',['swa448_sink4'])]:
   for kind in kinds:
    names=[kind] if kind in ['teacher','swa448_sink4'] else [f'{kind}_s{s}' for s in [11,29,47]];files=[R/f'ppl_{split}_{mode}_{n}.json' for n in names]
    if not all(f.exists() for f in files):continue
    vals=[read(f)['ppl'] for f in files];ppl.append(dict(split=split,mode=mode,kind=kind,ppl_mean=float(np.mean(vals)),ppl_seed_values=vals))
  pairs=[('pure','split_01','pure','split_kl'),('pure','split_01','pure','exp_kl'),('pure','split_01','pure','softplus_kl'),('hybrid','calibrated','hybrid','gate'),('hybrid','split_01','hybrid','softplus_kl'),('hybrid','split_01','window','swa448_sink4'),('hybrid','calibrated','window','swa448_sink4')]
  for ma,ka,mb,kb in pairs:
   def get(mode,kind):
    files=[R/f'ppl_{split}_{mode}_{kind}_s{s}.json' for s in [11,29,47]] if kind!='swa448_sink4' else [R/f'ppl_{split}_{mode}_{kind}.json']*3
    return None if not all(f.exists() for f in files) else [[r['nll']/r['tokens'] for r in read(f)['documents']] for f in files]
   a=get(ma,ka);b=get(mb,kb)
   if a is not None and b is not None:comparisons.append(dict(split=split,a=ma+'/'+ka,b=mb+'/'+kb,metric='delta NLL per token (negative favors a)',**paired_ci(a,b)))
 out=dict(kernel=kernel,ppl=ppl,paired_comparisons=comparisons);save(R/'summary.json',out)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 methods=['split_01','split_kl','exp_kl','softplus_kl','hedgehog','learned_prf','favor'];labels=['Raw mass + KL','Same-net KL','Exp MLP KL','Softplus MLP KL','Hedgehog form','Learned PRF','FAVOR+']
 fig,axs=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
 for ax,split in zip(axs,['confirm_wiki','confirm_long']):
  values=[next(r for r in kernel if r['split']==split and r['kind']==k) for k in methods];means=[r['output_nmse'] for r in values];lo=[min(r['output_nmse_seed_means']) for r in values];hi=[max(r['output_nmse_seed_means']) for r in values];ax.bar(np.arange(len(methods)),means,color=['#2166ac']+['#7899b3']*3+['#aaa']*3);ax.errorbar(np.arange(len(methods)),means,yerr=[np.array(means)-lo,np.array(hi)-means],fmt='none',color='black',capsize=3);ax.set_xticks(np.arange(len(methods)),labels,rotation=35,ha='right');ax.set_ylabel('Attention output NMSE (lower is better)');ax.set_title('Heldout Wiki, '+('1024 tokens' if split=='confirm_wiki' else '8192 tokens'))
 fig.savefig(P/'figures/kernel_generalization.pdf');fig.savefig(P/'figures/kernel_generalization.png',dpi=180);plt.close(fig)
 if len(ppl)>20:
  fig,axs=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
  choices=[('pure','teacher'),('window','swa448_sink4'),('pure','split_01'),('pure','exp_kl'),('pure','softplus_kl'),('hybrid','split_01'),('hybrid','softplus_kl'),('hybrid','calibrated'),('hybrid','gate')]
  for ax,split in zip(axs,['confirm_wiki','confirm_long']):
   use=[(m,k,next((r for r in ppl if r['mode']==m and r['kind']==k and r['split']==split),None)) for m,k in choices];use=[r for r in use if r[2] is not None];x=np.arange(len(use));vals=[r[2]['ppl_mean'] for r in use];ax.scatter(x,vals,color=['black' if m=='pure' and k=='teacher' else '#7570b3' if m=='window' else '#1b9e77' if m=='hybrid' else '#d95f02' for m,k,_ in use]);ax.set_xticks(x,[m+' / '+k for m,k,_ in use],rotation=40,ha='right');ax.set_ylabel('Full-model PPL');ax.set_title('Wiki, '+('1024' if split=='confirm_wiki' else '8192')+' tokens; 24/336 heads replaced')
  fig.savefig(P/'figures/full_model_ppl.pdf');fig.savefig(P/'figures/full_model_ppl.png',dpi=180);plt.close(fig)
 if (R/'benchmark.json').exists():
  bench=read(R/'benchmark.json')['results'];fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
  for mode,name in [('pure','teacher'),('pure','favor_s11'),('pure','split_01_s11'),('hybrid','calibrated_s11'),('window','swa448_sink4')]:
   rr=sorted([r for r in bench if r['mode']==mode and r['name']==name],key=lambda r:r['prompt_tokens']);xs=[r['prompt_tokens'] for r in rr]
   if rr:axs[0].plot(xs,[r['decode_ms_per_token'] for r in rr],'o-',label=mode+'/'+name);axs[1].plot(xs,[r['total_bytes']/2**20 for r in rr],'o-',label=mode+'/'+name)
  axs[0].set_ylabel('Full-model decode ms/token');axs[1].set_ylabel('Actual KV + linear state + window (MiB)')
  for ax in axs:ax.set_xlabel('Prompt tokens');ax.legend(fontsize=8)
  fig.savefig(P/'figures/inference.pdf');fig.savefig(P/'figures/inference.png',dpi=180);plt.close(fig)
 print(json.dumps(dict(kernel_rows=len(kernel),ppl_rows=len(ppl),paired_comparisons=len(comparisons))),flush=True)

if __name__=='__main__':main()
