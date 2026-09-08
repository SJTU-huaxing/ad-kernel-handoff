"""Aggregate matched experiments, audit identities, and export figures/CSVs."""
import csv,json,math,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent;OP=P.parent/'distribution_operator'
HEADS=[[14,0],[14,6],[27,0],[27,6]]
METHODS=['favor_plus','centered_favor_plus','sderf','aderf','partition','galerkin']
LABELS={'favor_plus':'FAVOR+','centered_favor_plus':'Centered FAVOR+', 'sderf':'SDERF','aderf':'ADERF',
        'partition':'New nonnegative partition','galerkin':'Train-Galerkin (signed)'}

def read(path):return json.loads(path.read_text())
def save(name,obj):(P/name).write_text(json.dumps(obj,indent=2,allow_nan=False))
def csvwrite(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (P/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

rows=[];energy_errors=[]
for ds in ['internal','official']:
    for h in HEADS:
        r=read(P/f'results/{ds}_L{h[0]}H{h[1]}.json')
        rows.extend(r['results']);energy_errors.append(r['reference_energy_relative_error'])
assert len(rows)==640
pooled=read(P/'results/pooled_640.json')['results'];assert len(pooled)==32
for r in pooled:
    base=[x['nmse'] for x in rows if x['dataset']==r['dataset'] and x['head']==r['head'] and x['method']==r['method'] and x['m']==128]
    assert r['nmse']<=statistics.mean(base)*(1+1e-8)
rows.extend(pooled)
g=read(OP/'results/train_galerkin_results.json')['results']
bounds={}
for ds in ['internal','official']:
    rr=read(OP/f'results/{ds}_summary.json')[-1];n=rr['samples_per_marginal']
    for h in rr['heads']:
        for m in [16,32,64,128]:
            bounds[(ds,tuple(h['head']),m)]=h['signed_floor_bracket'][str(m)]['lower']
            for method,error in [('partition',h['positive'][str(m)]['nmse']),
                                 ('galerkin',next(r['nmse'] for r in g if r['dataset']==ds and r['n']==n and r['head']==h['head'] and r['m']==m))]:
                rows.append(dict(dataset=ds,n=n,head=h['head'],method=method,seed=None,m=m,nmse=error,log_nmse=math.log(error)))
assert len(rows)==736
for r in rows:
    assert math.isfinite(r['nmse']) and r['nmse']>=0
    if r['m']<=128:assert r['nmse']+1e-8>=bounds[(r['dataset'],tuple(r['head']),r['m'])]
csvwrite('kernel_curves.csv',rows)

groups=[]
for ds in ['internal','official']:
    for head in HEADS:
        for method in METHODS:
            for m in [16,32,64,128,640]:
                a=[r['nmse'] for r in rows if r['dataset']==ds and r['head']==head and r['method']==method and r['m']==m]
                if not a:continue
                groups.append(dict(dataset=ds,head=head,method=method,m=m,realizations=len(a),
                    mean=statistics.mean(a),median=statistics.median(a),minimum=min(a),maximum=max(a),
                    std=statistics.stdev(a) if len(a)>1 else None))
csvwrite('kernel_summary.csv',groups)

attention=[]
for ds in ['internal','official']:
    attention.extend([dict(dataset=ds,**r) for r in read(P/f'results/{ds}_same_document.json')['results']])
assert len(attention)==208
csvwrite('attention_curves.csv',attention)

bench=read(P/'results/benchmark.json');assert len(bench['results'])==168
bb=[]
for r in bench['results']:
    flat={k:v for k,v in r.items() if k!='timing'}
    for k,v in r['timing'].items():flat[k+'_median_ms']=v['median_ms']
    bb.append(flat)
csvwrite('benchmark.csv',bb)
checks=read(P/'checks/implementation.json');precision=read(P/'checks/precision.json')
win={}
for ds in ['internal','official']:
    new={tuple(h):next(r['nmse'] for r in rows if r['dataset']==ds and r['method']=='partition' and r['head']==h and r['m']==64) for h in HEADS}
    win[ds]={method:dict(heads_better_than_median=sum(new[tuple(h)]<next(g['median'] for g in groups if g['dataset']==ds and g['head']==h and g['method']==method and g['m']==64) for h in HEADS),
                          heads_better_than_all_five_seeds=sum(new[tuple(h)]<next(g['minimum'] for g in groups if g['dataset']==ds and g['head']==h and g['method']==method and g['m']==64) for h in HEADS))
             for method in METHODS[:4]}
audit=dict(kernel_rows=736,rf_seed_rows=640,pooled_convexity_checks=32,signed_rank_lower_bound_checks=704,
           attention_rows=208,benchmark_configurations=168,max_reference_energy_error=max(energy_errors),
           implementation_checks=checks,pooled_direct_prediction_error=precision['max_pooled_direct_prediction_relative_error'],
           passed=True)
save('checks/aggregate.json',audit)
save('summary.json',dict(kernel_groups=groups,partition_m64_wins=win,audit=audit,
      interpretation='Matched fixed-Q/K empirical experiments. Five seeds for RF m<=128; one predefined five-block union for m=640. No claim about all LLMs or full-model throughput.'))

figdir=P/'figures';figdir.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130,'savefig.bbox':'tight'})
colors=['#687d9c','#9979b3','#bf975b','#b58d99','#cf6327','#18786a']
fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
for ax,h in zip(axes.flat,HEADS):
    for method,color in zip(METHODS,colors):
        rr=[r for r in groups if r['dataset']=='internal' and r['head']==h and r['method']==method and r['m']<=128]
        ax.plot([r['m'] for r in rr],[r['median'] for r in rr],'-o',color=color,label=LABELS[method],lw=2 if method in ['partition','galerkin'] else 1.2,markersize=4)
    ax.set_title(f'L{h[0]}H{h[1]}');ax.set_xscale('log',base=2);ax.set_yscale('log')
    ax.set_xticks([16,32,64,128],labels=[16,32,64,128]);ax.grid(alpha=.15)
for ax in axes[-1]:ax.set_xlabel('Feature dimension m')
for ax in axes[:,0]:ax.set_ylabel('Raw-kernel relative squared error')
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9)
fig.suptitle('Matched full held-out product: 131,072 Q x 131,072 K per head\nRandom features: median of five seeds; frozen operator constructions: one fit',fontsize=12)
fig.tight_layout(rect=(0,.09,1,.91));fig.savefig(figdir/'matched_kernel_curves.png');fig.savefig(figdir/'matched_kernel_curves.pdf');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4.8))
order=[('favor_plus',64),('sderf',64),('aderf',64),('partition',64),('galerkin',64),('favor_plus',640)]
tick=['FAVOR+\n64','SDERF\n64','ADERF\n64','Partition\n64','Galerkin\n64','FAVOR+\n640']
for ax,dtype in zip(axes,['float32','float64']):
    rs=[next(r for r in bb if r['method']==method and r['m']==m and r['n']==2048 and r['dtype']==dtype) for method,m in order]
    bars=ax.bar(np.arange(len(rs)),[r['total_median_ms'] for r in rs],color=['#687d9c','#bf975b','#b58d99','#cf6327','#18786a','#687d9c'])
    ax.bar_label(bars,fmt='%.2f',fontsize=9,padding=3)
    ax.set_xticks(np.arange(len(rs)),tick);ax.set_ylabel('Measured feature + aggregation latency (ms)')
    ax.set_title(dtype.upper());ax.set_ylim(0,max(r['total_median_ms'] for r in rs)*1.2);ax.grid(axis='y',alpha=.15)
fig.suptitle('RTX 3090: four heads, Nq=Nk=2048, d=dv=128\nRectangular inference microbenchmark; not end-to-end LLM throughput',fontsize=12)
fig.tight_layout(rect=(0,0,1,.88));fig.savefig(figdir/'latency.png');fig.savefig(figdir/'latency.pdf');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4.8))
for ax,ds in zip(axes,['internal','official']):
    for method,color in [('favor_plus','#687d9c'),('partition','#cf6327'),('galerkin','#18786a')]:
        vals=[statistics.median(r['output_nmse'] for r in attention if r['dataset']==ds and r['method']==method and r['m']==64 and r['head']==h) for h in HEADS]
        ax.plot(range(4),vals,'o-',color=color,label=LABELS[method])
    ax.set_xticks(range(4),[f'L{h[0]}H{h[1]}' for h in HEADS]);ax.set_yscale('log');ax.grid(alpha=.15)
    ax.set_title(ds);ax.set_ylabel('Attention output relative squared error')
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9)
fig.suptitle('Same-document 512-query x 512-key blocks, m=64\nSigned kernel accuracy does not guarantee a stable normalization denominator',fontsize=12)
fig.tight_layout(rect=(0,.1,1,.88));fig.savefig(figdir/'attention_outputs.png');fig.savefig(figdir/'attention_outputs.pdf');plt.close(fig)
print(json.dumps(dict(audit={k:v for k,v in audit.items() if k!='implementation_checks'},wins=win),indent=2))
