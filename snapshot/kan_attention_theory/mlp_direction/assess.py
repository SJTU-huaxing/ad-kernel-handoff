"""Complete frozen checks, new holdouts, rank selection and model PPL."""
import argparse,itertools
from core import *

def scenarios():
    # Final development ablation before any fresh activation extraction/evaluation.
    # Motivated by the observed KL-control vs calibrated-loss gap on validation.
    import subprocess
    if not all((P/'fits'/f'{kind}_m64_s{s}_n4096.json').exists() for kind in ['factorized','factorized_both','exp_control'] for s in [11,29,47]):
        subprocess.run([sys.executable,str(P/'train.py'),'--kinds','factorized','factorized_both','exp_control','--seeds','11','29','47'],check=True)
    selection=json.loads((P/'results/selection.json').read_text());winner=selection['winner']
    ranks=[16,32,64,96,128]
    table={m:json.loads((P/'fits'/f'{winner}_m{m}_s11_n4096.json').read_text())['validation']['summary']['output_nmse'] for m in ranks}
    configs=[ms for ms in itertools.product(ranks,repeat=4) if sum(ms)==256]
    valbest=min(configs,key=lambda ms:sum(table[m][h] for h,m in enumerate(ms)))
    spectra=json.loads((P/'results/spectrum.json').read_text())
    allocation={'uniform':[64]*4,'validation':list(valbest),**{f'spectrum_{k}':v for k,v in spectra['allocations'].items()}}
    cases={}
    for kind in sorted(set(['raw','balanced',winner,'value','kl_control','factorized','factorized_both','exp_control'])):
        for seed in [11,29,47]:cases[f'{kind}_m64_s{seed}']=[f'{kind}_m64_s{seed}_n4096']*4
    for rule,ms in allocation.items():cases[f'allocate_{rule}']=[f'{winner}_m{m}_s11_n4096' for m in ms]
    save(P/'results/scenarios.json',dict(cases=cases,allocation=allocation,validation_output_curves=table,
        source='Initial configurations frozen before first confirmation. Factorized query variant added after automatic round-1 evaluation, before round-2 fresh_wiki2/SWDE confirmation; exact total rank 256 and 295424 active Q/K network parameters.'))
    return cases

@torch.inference_mode()
def kernels():
    selection=json.loads((P/'results/selection.json').read_text());winner=selection['winner']
    names=[p.stem for p in sorted((P/'fits').glob('*.pt'))]
    results=[];path=P/'results/kernel.json'
    if path.exists():results=json.loads(path.read_text())['results']
    for ds in ['test','official','fresh_wiki','fda','fresh_wiki2','swde']:
        if ds in ['fresh_wiki2','swde'] and not (P/'results'/f'data_{ds}.pt').exists():continue
        data=load_full(ds) if ds in ['test','official'] else torch.load(P/'results'/f'data_{ds}.pt',weights_only=True)
        for name in names:
            if any(r['name']==name and r['dataset']==ds for r in results):continue
            model,meta=load_fit(name);metrics=evaluate(model,data,torch.tensor(meta['log_scale'],device='cuda'))
            row=dict(name=name,dataset=ds,metadata=meta,**metrics);results.append(row)
            save(path,dict(results=results,protocol='All heldout 64Q x 512K legal rectangles per document; FP64; no test fit.'))
            print(json.dumps(dict(event='kernel',name=name,dataset=ds,output=metrics['summary']['output_nmse'],raw=metrics['summary']['raw_nmse'])),flush=True)

@torch.inference_mode()
def ppl(gauge=False):
    from runtime_new import Replacement
    from evaluate_model import documents,load_model
    from transformers.models.qwen2 import modeling_qwen2
    config=json.loads((P/'results'/('gauge_plan.json' if gauge else 'scenarios.json')).read_text());cases=config['cases']
    if gauge:cases={k:v for k,v in cases.items() if not k.startswith('global_')}
    manifest,sets=documents()
    if gauge:sets={}
    for ds in (['fresh_wiki3','swde3'] if gauge else ['fresh_wiki','fda','fresh_wiki2','swde']):
        if (P/'results'/f'data_{ds}.pt').exists():sets[ds]=list(torch.load(P/'results'/f'data_{ds}.pt',weights_only=True)['input_ids'])
    original=modeling_qwen2.ALL_ATTENTION_FUNCTIONS['sdpa'];model=load_model(manifest)
    results=[];path=P/'results'/('ppl_gauge.json' if gauge else 'ppl.json')
    if path.exists():results=json.loads(path.read_text())['results']
    # Teacher and old single-pair MLP are independently measured on fresh datasets.
    cases={'teacher':None,'old_mlp':['old_mlp'],'favor_plus':['favor_plus'],**cases}
    unique={}
    for label,names in cases.items():
        key=tuple(names) if names is not None else None
        if key in unique:
            source=unique[key]
            for ds in sets:
                if not any(r['case']==label and r['dataset']==ds for r in results):
                    old=next(r for r in results if r['case']==source and r['dataset']==ds)
                    results.append(dict(old,case=label,identical_to=source))
            save(path,dict(results=results,scope='Frozen Qwen; exactly four heads; variable ranks have separate states, no padding.'))
            continue
        unique[key]=label
        if label in ['old_mlp','favor_plus']:
            if label=='old_mlp':
                sys.path.insert(0,str(ROOT/'candidate_validation'))
                from deploy import make_runtime
                runtime=make_runtime(original,'nn_mlp_11')
            else:
                from runtime import AttentionReplacement
                runtime=AttentionReplacement(original,'favor_plus',1009)
            runtime.collect_diagnostics=True
        else:runtime=Replacement(original,names) if names else None
        modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',runtime if runtime else original)
        for ds,docs in sets.items():
            if any(r['case']==label and r['dataset']==ds for r in results):continue
            rows=[];start=time.perf_counter()
            for i,doc in enumerate(docs):
                if runtime:runtime.reset()
                ids=doc.cuda();hidden=model.model(input_ids=ids[None],use_cache=False).last_hidden_state[0,:-1]
                losses=[]
                for at in range(0,len(hidden),128):
                    logits=model.lm_head(hidden[at:at+128]).float()
                    losses.append(F.cross_entropy(logits,ids[at+1:at+1+len(logits)],reduction='none'))
                losses=torch.cat(losses);assert torch.isfinite(losses).all()
                counts=torch.stack(runtime.diagnostics).sum(0).tolist() if runtime else [0,0,0]
                rows.append(dict(ordinal=i,nll=float(losses.mean()),first512_nll=float(losses[:511].mean()),last512_nll=float(losses[-512:].mean()),tokens=len(losses),nonpositive=counts[0],nonfinite=counts[1]))
            nll=sum(r['nll']*r['tokens'] for r in rows)/sum(r['tokens'] for r in rows)
            row=dict(case=label,dataset=ds,nll=nll,ppl=math.exp(nll),documents=rows,seconds=time.perf_counter()-start,names=names)
            results.append(row);save(path,dict(results=results,scope='Frozen Qwen; exactly four heads; variable ranks have separate states, no padding.'))
            print(json.dumps(dict(event='ppl',case=label,dataset=ds,ppl=row['ppl'],seconds=row['seconds'])),flush=True)
    modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['scenarios','kernels','ppl']);a=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    dict(scenarios=scenarios,kernels=kernels,ppl=ppl)[a.action]()
