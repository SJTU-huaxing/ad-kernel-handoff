import json
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parent;ROOT=P.parent;OLD=ROOT/'query_amplitude_ablation'
SEEDS=[11,29,47]
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def groups():
    out={'standard_ad':[f'causal_kl_reduced_plain_s{s}_lr0.002' for s in SEEDS]}
    for g,lr in read(P/'results/plan.json')['learning_rates'].items():out[g]=[f'{g}_s{s}_lr{lr:g}' for s in SEEDS]
    return out
def path(name,prefix):
    root=OLD if name.startswith('causal_kl_reduced_plain') else P
    return root/('fits' if prefix=='fit' else 'results')/(f'{name}.json' if prefix=='fit' else f'{prefix}_{name}.json')
def paired(exp,ad):
    delta=np.asarray(exp)-np.asarray(ad);assert delta.ndim==2 and delta.shape[0]==3
    dm=delta.mean(0);rng=np.random.default_rng(984257)
    ix=rng.integers(0,len(dm),size=(10000,len(dm)))
    return dict(exp_minus_ad=float(delta.mean()),seed_differences=delta.mean(-1).tolist(),
        conditional_document_ci95=np.quantile(dm[ix].mean(-1),[.025,.975]).tolist(),
        ad_better_documents=int((dm>0).sum()),documents=len(dm))

def main():
    gg=groups();table=[];comparisons=[];head_rows=[];training=[];selection=[]
    for g,names in gg.items():
        fits=[read(path(name,'fit')) for name in names]
        training.append(dict(group=g,lr=fits[0]['metadata']['lr'],
            seeds=SEEDS,training_seconds=[f['metadata']['training_seconds'] for f in fits],
            validation_kl=[f['selection_score'] for f in fits],
            steps=[r['step'] for r in fits[0]['history']],
            seed_mean_head_curves=[[np.mean(r['per_head_loss']).item() for r in f['history']] for f in fits]))
        for lr in [.002,.0005]:
            name=f'causal_kl_reduced_plain_s11_lr{lr:g}' if g=='standard_ad' else f'{g}_s11_lr{lr:g}'
            f=read(path(name,'fit'))
            selection.append(dict(group=g,lr=lr,validation_kl=f['selection_score'],selected=lr==fits[0]['metadata']['lr']))
    for split in ['confirm_wiki','confirm_long']:
        kk={g:[read(path(n,f'kernel_{split}')) for n in names] for g,names in gg.items()}
        pp={g:[read(path(n,f'ppl_{split}')) for n in names] for g,names in gg.items()}
        dd=read(P/'results'/f'diagnostics_{split}.json')['models']
        for g,names in gg.items():
            kl=np.asarray([x['summary']['kl'] for x in kk[g]]);nm=np.asarray([x['summary']['output_nmse'] for x in kk[g]])
            dr=[row for name in names for row in dd[name]]
            row=dict(split=split,group=g,kl=float(kl.mean()),output_nmse=float(nm.mean()),
                ppl=float(np.mean([x['ppl'] for x in pp[g]])),seed_ppl=[x['ppl'] for x in pp[g]],
                seed_kl=kl.mean(-1).tolist(),seed_output_nmse=nm.mean(-1).tolist(),
                layer_kl=kl.mean(0).reshape(2,12).mean(-1).tolist())
            for field in ['head_tv','head_coeff_l2','layer_projected_nmse']:
                row[field]=float(np.mean([r[field] for r in dr]))
            for prefix in ['head_output','layer_projected']:
                a=np.asarray([r[prefix+'_sse'] for r in dr]);b=np.asarray([r[prefix+'_energy'] for r in dr])
                row[prefix+'_global_nmse']=float(a.sum()/b.sum())
            if split=='confirm_long':
                for field in ['severe_underestimate_mass','severe_positive_kl']:
                    row[field]=float(np.mean([r[field] for r in dr]))
            row['positions']=[]
            for start in [0,1024,2048,4096]:
                pr=[v for r in dr for v in r['positions'] if v['start']==start]
                if pr:
                    nq=sum(v['queries'] for v in pr)
                    row['positions'].append(dict(start=start,end=pr[0]['end'],
                        kl=float(np.sum([v['kl'] for v in pr])/(nq*24)),
                        tv=float(np.sum([v['tv'] for v in pr])/(nq*24))))
            table.append(row)
            for idx,head in enumerate([[l,h] for l in [14,27] for h in range(12)]):
                head_rows.append(dict(group=g,split=split,head=head,kl=float(kl[:,idx].mean()),output_nmse=float(nm[:,idx].mean())))
        for init in ['standard','matched_zero']:
            a,b=f'{init}_exp',f'{init}_ad'
            for metric in ['kl','output_nmse','nll','head_tv','head_coeff_l2','layer_projected_nmse']:
                def values(g):
                    if metric=='nll':return [[r['nll']/r['tokens'] for r in x['documents']] for x in pp[g]]
                    if metric in ['kl','output_nmse']:return [[np.mean(r[metric]) for r in x['documents']] for x in kk[g]]
                    return [[np.mean(r[metric]) for r in dd[n]] for n in gg[g]]
                comp=dict(split=split,initialization=init,metric=metric,**paired(values(a),values(b)))
                if metric in ['kl','output_nmse']:
                    delta=np.mean([x['summary'][metric] for x in kk[a]],axis=0)-np.mean([x['summary'][metric] for x in kk[b]],axis=0)
                    comp['ad_better_heads']=int((delta>0).sum());comp['per_head_difference']=delta.tolist()
                comparisons.append(comp)
    teacher_check=[]
    for split in ['confirm_wiki','confirm_long']:
        new=read(P/'results'/f'ppl_{split}_teacher.json');old=read(OLD/'results'/f'ppl_{split}_teacher.json')
        teacher_check.append(dict(split=split,new=new['ppl'],old=old['ppl'],difference=new['ppl']-old['ppl']))
    save(P/'results/summary.json',dict(table=table,comparisons=comparisons,heads=head_rows,training=training,
        selection=selection,teacher_check=teacher_check,benchmark=read(P/'results/benchmark.json')['aggregates'],
        scope='EXP minus AD; positive difference favors AD. Conditional paired document bootstrap across3fixed seeds,10000resamples; not independent model-family or training-seed confidence interval. Existing heldout was not used for this run hyperparameter selection.'))
    print(json.dumps(dict(table=table,comparisons=[{k:v for k,v in c.items() if k!='per_head_difference'} for c in comparisons],selection=selection,teacher_check=teacher_check),indent=2))

if __name__=='__main__':main()
