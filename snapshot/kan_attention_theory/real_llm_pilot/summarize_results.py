"""Aggregate document-level errors; produce reviewable tables, checks and figures."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=Path(__file__).resolve().parent;out=p/'results';out.mkdir(exist_ok=True)
manifest=json.loads((p/'data_qwen25_1p5b/manifest.json').read_text())
spectra=json.loads((p/'analysis_qwen25_1p5b/spectra.json').read_text())
sink=json.loads((p/'extra/sink_spectra.json').read_text())
baseline=json.loads((p/'extra/baselines.json').read_text())
fits=[]
for folder in ['fits','fits_dimensions','fits_longer']:
    for file in sorted((p/folder).glob('*.json')):
        row=json.loads(file.read_text());row['folder']=folder;fits.append(row)
groups={}
for row in fits:
    key=(row['folder'],row['objective'],row['method'],row['m'],row['steps'])
    groups.setdefault(key,[]).append(row)
metrics=['kernel_nmse','attention_nmse','output_nmse','row_l1','negative_fraction',
         'nonpositive_denominator_fraction','top1_agreement']
table=[]
for key,runs in groups.items():
    for split in runs[0]['evaluations']:
        for metric in metrics:
            values=np.array([[r[metric] for r in run['evaluations'][split]] for run in runs])
            perseed=values.mean(1)
            for h,(layer,head) in enumerate(runs[0]['head_labels']):
                table.append(dict(folder=key[0],objective=key[1],method=key[2],m=key[3],steps=key[4],
                    split=split,metric=metric,layer=layer,head=head,seeds=len(runs),documents=values.shape[1],
                    mean=float(perseed[:,h].mean()),seed_std=float(perseed[:,h].std(ddof=1)) if len(runs)>1 else 0,
                    parameters=runs[0]['parameters_per_head_pair'],
                    training_seconds=float(np.mean([r['training_seconds'] for r in runs])),
                    feature_pair_ms=float(np.mean([r['feature_pair_512tokens_ms'] for r in runs]))))
with (out/'metrics.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)

labels=[[l,h] for l in [0,14,27] for h in [0,6]]
names=[f'L{l} H{h}' for l,h in labels]
checks=dict(unique_document_hashes=len({r['text_sha256'] for r in manifest['documents']})==len(manifest['documents']),
            unique_token_hashes=len({r['token_sha256'] for r in manifest['documents']})==len(manifest['documents']),
            fit_groups=len(groups),fit_runs=len(fits),svd_floor_comparisons=0,svd_floor_violations=[])
floor={(r['document'],r['layer'],r['head']):r['floors'] for r in spectra
       if r['source']=='same_context_legal_512x512_block' and r['split']=='test' and r['kind']=='row_normalized'}
for run in fits:
    for row in run['evaluations']['test_block']:
        for h,(layer,head) in enumerate(run['head_labels']):
            f=floor.get((row['file'],layer,head))
            if f:
                checks['svd_floor_comparisons']+=1
                if row['attention_nmse'][h]+1e-5<f[str(run['m'])]:
                    checks['svd_floor_violations'].append([run['name'],row['file'],layer,head])

comparisons=[];rng=np.random.default_rng(20260905)
for folder in ['fits','fits_longer']:
    runs={method:sorted([r for r in fits if r['folder']==folder and r['objective']=='row' and r['m']==64
                        and r['method']==method],key=lambda x:x['seed']) for method in ['kan_positive','mlp_positive']}
    if min(map(len,runs.values()))!=3:continue
    for split in ['test','test_long','ood']:
        for metric in ['attention_nmse','output_nmse']:
            values={method:np.array([[row[metric] for row in r['evaluations'][split]] for r in rs])
                    for method,rs in runs.items()}
            diff=(values['kan_positive']-values['mlp_positive']).mean(0)
            boot=diff[rng.integers(len(diff),size=(5000,len(diff)))].mean(1)
            for h,label in enumerate(labels):
                comparisons.append(dict(folder=folder,split=split,metric=metric,head_label=label,
                    kan_minus_mlp=float(diff[:,h].mean()),
                    document_bootstrap_ci95=np.quantile(boot[:,h],[.025,.975]).tolist(),
                    note='Paired document bootstrap, conditional on the three trained seeds; exploratory, not multiplicity corrected.'))

replacement_path=p/'replacement/replacement.json'
replacement=json.loads(replacement_path.read_text()) if replacement_path.exists() else None
summary=dict(checks=checks,metrics=table,paired_differences=comparisons,replacement=replacement)
(out/'summary.json').write_text(json.dumps(summary,indent=2))

def get(method,metric='attention_nmse',split='test',folder='fits',m=64,objective='row'):
    rows=[r for r in table if r['folder']==folder and r['method']==method and r['metric']==metric
          and r['split']==split and r['m']==m and r['objective']==objective]
    lookup={(r['layer'],r['head']):r for r in rows}
    return np.array([lookup.get(tuple(label),{}).get('mean',np.nan) for label in labels]),np.array(
        [lookup.get(tuple(label),{}).get('seed_std',np.nan) for label in labels])

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
colors={'mlp_positive':'#2878b5','kan_positive':'#d95319'}
x=np.arange(6);fig,axes=plt.subplots(2,2,figsize=(12,8.4))
for ax,metric,title in [(axes[0,0],'attention_nmse','Normalized attention error'),(axes[0,1],'output_nmse','Head output error')]:
    for method,offset in [('mlp_positive',-.18),('kan_positive',.18)]:
        mean,std=get(method,metric);ax.bar(x+offset,mean,.34,yerr=std,capsize=3,
            label=method.replace('_positive',' + softplus').upper(),color=colors[method])
    ax.set_xticks(x,names);ax.set_ylabel('Relative squared error');ax.set_title(title+' (2,000 steps)');ax.legend(fontsize=8)
ax=axes[1,0]
for skip,label,style in [(0,'All 512 prefix keys','o-'),(1,'Remove first key','s--'),(4,'Remove first 4 keys','^:')]:
    vals=[]
    for layer,head in labels:
        rows=([r for r in spectra if r['split']=='test' and r['source']=='same_context_legal_512x512_block']
              if skip==0 else [r for r in sink if r['removed_prefix']==skip])
        vals.append(np.mean([r['floors']['64'] for r in rows if r['layer']==layer and r['head']==head and r['kind']=='row_normalized']))
    ax.semilogy(x,vals,style,label=label)
ax.set_xticks(x,names);ax.set_ylabel('Best rank-64 relative squared error');ax.set_title('Empirical SVD floors; re-normalize remaining keys');ax.legend(fontsize=8)
ax=axes[1,1]
for method,offset in [('mlp_positive',-.18),('kan_positive',.18)]:
    mean,std=get(method,folder='fits_longer');ax.bar(x+offset,mean,.34,yerr=std,capsize=3,label=method,color=colors[method])
ax.set_xticks(x,names);ax.set_ylabel('Relative squared error');ax.set_title('Normalized attention error (10,000 steps)')
fig.suptitle('Frozen Qwen2.5-1.5B: independent Q/K maps, m=64, 3 seeds',fontsize=14)
fig.tight_layout();fig.savefig(out/'pilot_overview.png');fig.savefig(out/'pilot_overview.pdf');plt.close(fig)

fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
for ax,i in zip(axes.flat,range(6)):
    for method in colors:
        means=[];stds=[]
        for m in [32,64,128]:
            a,b=get(method,folder='fits' if m==64 else 'fits_dimensions',m=m);means.append(a[i]);stds.append(b[i])
        ax.errorbar([32,64,128],means,yerr=stds,marker='o',capsize=3,label=method,color=colors[method])
    ax.set_title(names[i]);ax.set_xticks([32,64,128]);ax.set_xlabel('Feature dimension m');ax.set_ylabel('Attention NMSE')
axes[0,0].legend(fontsize=8);fig.suptitle('Dimension sweep: same 2,000 steps, parameter-matched within each m')
fig.tight_layout();fig.savefig(out/'dimension_sweep.png');plt.close(fig)

if replacement:
    rows=replacement['results'];teacher=next(r for r in rows if r['scenario']=='teacher')
    fig,ax=plt.subplots(figsize=(9,4.5));scopes=['middle','late','four_heads','six_heads']
    for method,offset in [('mlp_positive',-.17),('kan_positive',.17)]:
        a=[];b=[]
        for scope in scopes:
            vals=[r['last512_nll']-teacher['last512_nll'] for r in rows if r['scenario'].startswith('row_'+method)
                  and r['scenario'].endswith('_'+scope)]
            a.append(np.mean(vals));b.append(np.std(vals,ddof=1))
        ax.bar(np.arange(4)+offset,a,.32,yerr=b,capsize=3,color=colors[method],label=method)
    ax.axhline(0,color='gray',lw=1);ax.set_xticks(range(4),['2 middle heads','2 late heads','4 middle/late','6 including first layer'])
    ax.set_ylabel('Increase in next-token NLL (nats)');ax.set_title('Full-LLM replacement: last 512 targets, 12 held-out documents')
    ax.legend();fig.tight_layout();fig.savefig(out/'model_replacement.png');plt.close(fig)
print(json.dumps(checks,indent=2))
