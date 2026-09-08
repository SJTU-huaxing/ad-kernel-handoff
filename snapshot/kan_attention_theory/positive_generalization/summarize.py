"""Aggregate by kernel energy, retaining both seed means and medians."""
import csv,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
LABELS=[[14,0],[14,6],[27,0],[27,6]];RANKS=[16,32,64,128]

def lse(x,axis=0):
    x=np.asarray(x);m=x.max(axis=axis,keepdims=True)
    return np.squeeze(m,axis=axis)+np.log(np.exp(x-m).sum(axis=axis))

def aggregate():
    rows=[]
    for dataset in ['internal','official']:
        for source in ['product','same_context']:
            blocks=[json.loads((P/f'results/{dataset}_{source}_{r}.json').read_text()) for r in range(4)]
            for rep,block in enumerate(blocks):
                extra=P/f'results/beta15_{dataset}_{source}_{rep}.json'
                if extra.exists():block['methods']+=json.loads(extra.read_text())['methods']
            energies=np.array([b['methods'][0]['log_energy'] for b in blocks]);total=lse(energies)
            methods=sorted({r['method'] for r in blocks[0]['methods']})
            for m in RANKS:
                for method in ['SVD','NMF']+methods:
                    if method.startswith('beta15') and m!=64:continue
                    if method in ['SVD','NMF']:
                        values=np.array([b['svd'][str(m)] if method=='SVD' else next(r['error'] for r in b['nmf'] if r['m']==m) for b in blocks])
                        scores=np.exp(lse(np.log(values.clip(1e-300))+energies)-total)[None];aux={}
                    else:
                        seeds=sorted({r['seed'] for r in blocks[0]['methods'] if r['method']==method});scores=[];aux={}
                        for seed in seeds:
                            data=[next(r for r in b['methods'] if r['method']==method and r['m']==m and r['seed']==seed) for b in blocks]
                            scores.append(np.exp(lse(np.array([r['log_nmse'] for r in data])+energies)-total))
                            mass=np.array([r['log_mass'] for r in data]);mass-=mass.max(0);weight=np.exp(mass);weight/=weight.sum(0)
                            for metric in ['idiv_per_mass','logcosh','factor2']+(['output_nmse','row_l1'] if source=='same_context' else []):
                                v=np.array([r[metric] for r in data]);val=(v*weight).sum(0) if metric=='idiv_per_mass' else v.mean(0)
                                aux.setdefault(metric,[]).append(val)
                        scores=np.array(scores)
                    for h,label in enumerate(LABELS):
                        row=dict(dataset=dataset,source=source,head=f'L{label[0]}H{label[1]}',head_index=h,m=m,method=method,
                            seed_mean=float(scores[:,h].mean()),seed_median=float(np.median(scores[:,h])),seed_min=float(scores[:,h].min()),seed_max=float(scores[:,h].max()),seed_count=len(scores))
                        for metric,value in aux.items():row[metric]=float(np.asarray(value)[:,h].mean())
                        rows.append(row)
    return rows

def get(rows,ds,src,m,method):return [r for r in rows if r['dataset']==ds and r['source']==src and r['m']==m and r['method']==method]

def main():
    rows=aggregate();columns=sorted({key for r in rows for key in r})
    with (P/'curves.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    fit=[json.loads(f.read_text()) for f in sorted((P/'fits').glob('*_m*_s*.json'))]
    assert len(fit)==24 and all(r['steps']==4096 and r['epochs']==1 and r['unique_pairs_per_head']==2097152 for r in fit)
    convergence=[]
    for kind in ['shared','untied']:
        rr=[r for r in fit if r['m']==64 and r['kind']==kind]
        for h,label in enumerate(LABELS):
            vals={step:float(np.mean([next(t['validation']['idiv_per_mass'][h] for t in r['history'] if t['step']==step) for r in rr])) for step in [0,2048,3584,4096]}
            convergence.append(dict(kind=kind,head=label,validation_idiv=vals,
                last_512_step_relative_improvement=(vals[3584]-vals[4096])/vals[3584],
                train_idiv=float(np.mean([r['train']['idiv_per_mass'][h] for r in rr])),
                test_idiv=float(np.mean([r['test']['idiv_per_mass'][h] for r in rr])),
                train_raw_nmse=float(np.mean([r['train']['raw_nmse'][h] for r in rr])),
                test_raw_nmse=float(np.mean([r['test']['raw_nmse'][h] for r in rr]))))
    gates=[]
    baselines=['favor_orf','sderf_orf','aderf_orf','balanced_orf','balanced_sderf_orf','balanced_sderf_sobol','balanced_sderf_sobol_mass']
    for ds in ['internal','official']:
        for src in ['product','same_context']:
            for kind in ['shared','untied']:
                learned=get(rows,ds,src,64,'learned_'+kind);nmf=get(rows,ds,src,64,'NMF')
                baseline=np.array([[r['seed_mean'] for r in get(rows,ds,src,64,b)] for b in baselines])
                error=np.array([r['seed_median'] for r in learned]);oracle=np.array([r['seed_mean'] for r in nmf])
                gates.append(dict(dataset=ds,source=src,kind=kind,
                    beats_every_rf_by_25pct=(error<=.75*baseline.min(0)).tolist(),
                    within_10x_nmf=(error<=10*oracle).tolist(),ratio_to_nmf=(error/oracle).tolist()))
    summary=dict(aggregated_rows=rows,convergence=convergence,gates=gates,fit_count=len(fit),
        training_seconds=sum(r['seconds'] for r in fit),parameters={r['name']:r['parameters_per_head'] for r in fit})
    beta=[json.loads(f.read_text()) for f in sorted((P/'beta15_fits').glob('*_m64_s*.json'))]
    summary['exploratory_beta15']=dict(fits=len(beta),training_seconds=sum(r['seconds'] for r in beta),
        note='Loss chosen after seeing initial test overshoot; no fresh confirmatory holdout.')
    (P/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    figures=P/'figures';figures.mkdir(exist_ok=True)
    colors={'SVD':'black','NMF':'gray','favor_orf':'#4978ad','aderf_orf':'#47905a','sderf_orf':'#85a92c',
            'balanced_sderf_sobol':'#d29b28','learned_shared':'#d3524e','learned_untied':'#9e64b5'}
    names={'SVD':'SVD oracle','NMF':'NMF (test matrix)','favor_orf':'FAVOR+','aderf_orf':'ADERF (FAVOR# family)',
           'sderf_orf':'SDERF (FAVOR# family)','balanced_sderf_sobol':'Balanced SDERF + Sobol',
           'learned_shared':'Learned shared nodes','learned_untied':'Learned Q/K nodes'}
    for ds in ['internal','official']:
        fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
        for h,(ax,label) in enumerate(zip(axes.flat,LABELS)):
            for method,color in colors.items():
                series=[get(rows,ds,'product',m,method)[h]['seed_mean'] for m in RANKS]
                ax.plot(RANKS,series,'o-',label=names[method],color=color,linewidth=1.5,markersize=3)
            ax.set_yscale('log');ax.set_xscale('log',base=2);ax.set_xticks(RANKS,labels=RANKS);ax.axhline(1,color='#777',linestyle=':',linewidth=.7)
            ax.set_title(f'L{label[0]} H{label[1]}');ax.set_xlabel('Positive feature dimension m');ax.set_ylabel('Raw kernel energy-relative squared error');ax.grid(alpha=.2)
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
        fig.suptitle(f'{ds.capitalize()} held-out documents | product marginals | seed means')
        fig.tight_layout(rect=(0,.10,1,.95));fig.savefig(figures/f'{ds}_generalization.png',dpi=180);fig.savefig(figures/f'{ds}_generalization.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,kind in zip(axes,['shared','untied']):
        rr=[r for r in fit if r['m']==64 and r['kind']==kind]
        steps=[t['step'] for t in rr[0]['history']]
        for h,label in enumerate(LABELS):
            curve=np.array([[t['validation']['idiv_per_mass'][h] for t in r['history']] for r in rr])
            ax.plot(steps,curve.mean(0),label=f'L{label[0]} H{label[1]}');ax.fill_between(steps,curve.min(0),curve.max(0),alpha=.10)
        ax.set_title(f'{kind.capitalize()} nodes, m=64');ax.set_xlabel('Single-pass updates (512 new pairs / update)');ax.set_ylabel('Validation raw I-divergence / target mass');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(figures/'validation_convergence.png',dpi=180);plt.close(fig)
    if beta and any(r['method']=='beta15_shared' for r in rows):
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        methods=['favor_orf','learned_shared','learned_untied','beta15_shared','beta15_untied']
        for ax,ds in zip(axes,['internal','official']):
            for j,method in enumerate(methods):
                yy=[r['seed_mean'] for r in get(rows,ds,'product',64,method)]
                ax.bar(np.arange(4)+(j-2)*.15,yy,width=.14,label=method)
            ax.set_xticks(range(4),labels=[f'L{l}H{h}' for l,h in LABELS]);ax.set_yscale('log');ax.axhline(1,color='gray',linestyle=':')
            ax.set_title(f'{ds.capitalize()} product holdout, m=64');ax.set_ylabel('Raw kernel relative squared error');ax.grid(axis='y',alpha=.2)
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
        fig.suptitle('Exploratory loss ablation: beta=1.5 after diagnosing overshoot')
        fig.tight_layout(rect=(0,.13,1,.95));fig.savefig(figures/'beta15_ablation.png',dpi=180);plt.close(fig)
    for ds in ['internal','official']:
        print(ds)
        for method in ['SVD','NMF','favor_orf','sderf_orf','aderf_orf','balanced_sderf_sobol','learned_shared','learned_untied']:
            print(method,[round(r['seed_mean'],6) for r in get(rows,ds,'product',64,method)])
    print('GATES',json.dumps(gates))

if __name__=='__main__':main()
