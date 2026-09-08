import numpy as np
from common_eval import *

def paired(a,b):
    delta=np.asarray(a)-np.asarray(b)
    assert delta.ndim==2
    dm=delta.mean(0)
    rng=np.random.default_rng(984257)
    ix=rng.integers(0,len(dm),size=(10000,len(dm)))
    return dict(difference=float(delta.mean()),seed_differences=delta.mean(-1).tolist(),
        conditional_document_ci95=np.quantile(dm[ix].mean(-1),[.025,.975]).tolist(),
        better_documents=int((dm<0).sum()),documents=len(dm))

def main():
    table=[];comparisons=[];heads=[];diagnostics=[]
    for split in ['confirm_wiki','confirm_long']:
        kk={g:[read(parent_file(g,seed,split,'kernel')) for seed in SEEDS] for g in GROUPS}
        pp={g:[read(parent_file(g,seed,split,'ppl')) for seed in SEEDS] for g in GROUPS}
        for g in GROUPS:
            kl=np.asarray([v['summary']['kl'] for v in kk[g]])
            nm=np.asarray([v['summary']['output_nmse'] for v in kk[g]])
            table.append(dict(split=split,group=g,kl=float(kl.mean()),output_nmse=float(nm.mean()),
                ppl=float(np.mean([v['ppl'] for v in pp[g]])),
                seed_ppl=[v['ppl'] for v in pp[g]],
                layer_kl=kl.mean(0).reshape(2,12).mean(-1).tolist(),
                layer_output_nmse=nm.mean(0).reshape(2,12).mean(-1).tolist()))
            for idx,head in enumerate(HEADS):
                heads.append(dict(split=split,group=g,head=head,kl=float(kl[:,idx].mean()),output_nmse=float(nm[:,idx].mean())))
        for other in GROUPS[1:]:
            for metric in ['kl','output_nmse','nll']:
                source_data=pp if metric=='nll' else kk
                def values(g):
                    return [[d['nll']/d['tokens'] if metric=='nll' else np.mean(d[metric])
                             for d in v['documents']] for v in source_data[g]]
                result=dict(split=split,a='ad_plain',b=other,metric=metric,**paired(values('ad_plain'),values(other)))
                if metric!='nll':
                    ah=np.asarray([v['summary'][metric] for v in kk['ad_plain']]).mean(0)
                    bh=np.asarray([v['summary'][metric] for v in kk[other]]).mean(0)
                    result['better_heads']=int((ah<bh).sum())
                comparisons.append(result)
        dp=P/'results'/f'diagnostics_{split}.json'
        if dp.exists():
            data=read(dp)['models']
            for g in GROUPS:
                rows=[d for seed in SEEDS for d in data[model_name(g,seed)]]
                diagnostic=dict(split=split,group=g)
                for field in ['head_tv','head_coeff_l2','layer_projected_nmse']:
                    a=np.asarray([d[field] for d in rows])
                    diagnostic[field]=float(a.mean());diagnostic[field+'_by_unit']=a.mean(0).tolist()
                for prefix in ['head_output','layer_projected']:
                    a=np.asarray([d[prefix+'_sse'] for d in rows]);b=np.asarray([d[prefix+'_energy'] for d in rows])
                    diagnostic[prefix+'_global_nmse']=float(a.sum()/b.sum())
                    diagnostic[prefix+'_global_nmse_by_unit']=(a.sum(0)/b.sum(0)).tolist()
                position=[]
                for start in [0,1024,2048,4096]:
                    pr=[v for d in rows for v in d['positions'] if v['start']==start]
                    if not pr:continue
                    n=sum(v['queries'] for v in pr)
                    position.append(dict(start=start,end=pr[0]['end'],queries_across_seeds=n,
                        kl=float(np.sum([v['kl'] for v in pr])/(n*24)),
                        tv=float(np.sum([v['tv'] for v in pr])/(n*24)),
                        output_global_nmse=float(np.sum([v['output_sse'] for v in pr])/np.sum([v['output_energy'] for v in pr]))))
                diagnostic['positions']=position;diagnostics.append(diagnostic)
            for other in GROUPS[1:]:
                for field in ['head_tv','head_coeff_l2','layer_projected_nmse']:
                    def values(g):
                        return [[np.mean(d[field]) for d in data[model_name(g,seed)]] for seed in SEEDS]
                    comparisons.append(dict(split=split,a='ad_plain',b=other,metric=field,
                        **paired(values('ad_plain'),values(other))))
    check=[]
    for split in ['confirm_wiki','confirm_long']:
        for seed in SEEDS:
            new=read(parent_file('favor',seed,split,'ppl'))
            old=read(ROOT/'causal_direction/results'/f'ppl_{split}_pure_favor_s{seed}.json')
            check.append(dict(split=split,seed=seed,new_ppl=new['ppl'],old_ppl=old['ppl'],difference=new['ppl']-old['ppl']))
    tail=[]
    tp=P/'results/tail_diagnostics.json'
    if tp.exists():
        data=read(tp)['models']
        for g in GROUPS:
            rows=[d for seed in SEEDS for d in data[model_name(g,seed)]]
            out=dict(group=g,positive_kl=float(np.mean([d['positive_kl'] for d in rows])),
                negative_kl=float(np.mean([d['negative_kl'] for d in rows])),thresholds=[])
            for i in range(4):
                out['thresholds'].append(dict(log_ratio_threshold=rows[0]['thresholds'][i]['log_ratio_threshold'],
                    **{k:float(np.mean([d['thresholds'][i][k] for d in rows])) for k in ['teacher_mass','prediction_mass','positive_kl','keys']}))
            tail.append(out)
    save(P/'results/summary.json',dict(table=table,comparisons=comparisons,heads=heads,diagnostics=diagnostics,tail=tail,favor_recheck=check,
        scope='Conditional document CIs on3fixed fits; same previously used heldout documents, no independent claim across model families.'))
    print(json.dumps(dict(table=table,comparisons=comparisons,diagnostics=diagnostics,favor_recheck=check),indent=2))

if __name__=='__main__':main()
