import argparse
from core import *

@torch.inference_mode()
def kernels():
    plan=json.loads((P/'results/gauge_plan.json').read_text());results=[]
    path=P/'results/kernel_gauge.json'
    if path.exists():results=json.loads(path.read_text())['results']
    for ds in ['fresh_wiki3','swde3']:
        data=torch.load(P/'results'/f'data_{ds}.pt',weights_only=True)
        for label,names in plan['cases'].items():
            if any(r['name']==names[0] and r['dataset']==ds for r in results):continue
            net,meta=load_fit(names[0]);scale=torch.tensor(meta['log_scale'],device='cuda')
            metrics=evaluate(net,data,scale);results.append(dict(name=names[0],dataset=ds,metadata=meta,**metrics))
            save(path,dict(results=results,protocol='Round-3 confirmation only; no fitting or model selection on these documents.'))
            print(json.dumps(dict(event='gauge_kernel',name=names[0],dataset=ds,balanced=metrics['summary']['balanced'],mass=metrics['summary']['mass'])),flush=True)
    checks=[]
    for ds in ['fresh_wiki3','swde3']:
        for seed in [11,29,47]:
            old=next(r for r in results if r['dataset']==ds and r['name']==f'exp_control_m64_s{seed}_n4096')
            new=next(r for r in results if r['dataset']==ds and r['name']==f'gauge_calibrated_m64_s{seed}_n4096')
            delta=max(abs(a-b) for x,y in zip(old['documents'],new['documents']) for a,b in zip(x['directional_kl'],y['directional_kl']))
            output_delta=max(abs(a-b) for x,y in zip(old['documents'],new['documents']) for a,b in zip(x['output_nmse'],y['output_nmse']))
            assert delta<1e-8 and output_delta<1e-8,(ds,seed,delta,output_delta)
            checks.append(dict(dataset=ds,seed=seed,max_document_directional_kl_change=delta,max_document_output_nmse_change=output_delta))
    save(P/'checks/gauge_invariance.json',checks)

def ppl():
    import assess
    assess.ppl(gauge=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['kernels','ppl']);a=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    (kernels if a.action=='kernels' else ppl)()
