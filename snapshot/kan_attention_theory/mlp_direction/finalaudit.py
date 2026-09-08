"""Audit measurements, calibration algebra, parameter budget and holdout identity."""
from core import *

@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    read=lambda n:json.loads((P/n).read_text())
    fits=[read('fits/'+p.name)['metadata'] for p in (P/'fits').glob('*.json')]
    assert len(fits)==38
    for meta in fits:
        net,_=load_fit(meta['name'])
        assert sum(p.numel() for p in net.parameters())==4*(49152+386*meta['m'])
        assert meta['training_pairs_per_head']==134217728 and meta['epochs']==1
        if meta['kind']=='gauge_calibrated':assert meta['calibration_pairs_per_head']==4294967296 and meta['calibration_queries_disjoint_from_direction_training']
    ppl=[]
    for f in ['results/ppl.json','results/ppl_gauge.json']:
        for r in read(f)['results']:
            docs=r['documents'];nll=sum(d['nll']*d['tokens'] for d in docs)/sum(d['tokens'] for d in docs)
            assert abs(nll-r['nll'])<1e-12 and abs(math.exp(nll)-r['ppl'])<1e-12
            assert all(d['nonpositive']==0 and d['nonfinite']==0 for d in docs)
            ppl.append(dict(file=f,case=r['case'],dataset=r['dataset'],documents=len(docs)))
    gp=read('results/ppl_gauge.json')['results'];deltas=[]
    for ds in ['fresh_wiki3','swde3']:
        for seed in [11,29,47]:
            a=next(r for r in gp if r['case']==f'exp_control_m64_s{seed}' and r['dataset']==ds)
            b=next(r for r in gp if r['case']==f'gauge_calibrated_m64_s{seed}' and r['dataset']==ds)
            deltas.append(dict(dataset=ds,seed=seed,nll_change=b['nll']-a['nll'],ppl_change=b['ppl']-a['ppl'],max_document_nll_change=max(abs(x['nll']-y['nll']) for x,y in zip(a['documents'],b['documents']))))
    records=[r for f in ['fresh_manifest.json','fresh_manifest2.json','fresh_manifest3.json'] for r in read('results/'+f)['records']]
    a={r['group'] for r in records if r['dataset']=='swde'};b={r['group'] for r in records if r['dataset']=='swde3'}
    assert not a&b and len(a)==96 and len(b)==40
    # Algebra on genuine unseen q/k: exact I-projection decomposition under
    # the empirical key reference, independent of the learned readout quality.
    data=torch.load(P/'results/data_fresh_wiki3.pt',weights_only=True)
    q=data['q'][0,:,512:549].cuda().double();k=data['k'][1,:,:61].cuda().double()
    net,meta=load_fit('exp_control_m64_s11_n4096');scale=torch.tensor(meta['log_scale'],device='cuda',dtype=torch.float64)
    t=(q@k.transpose(-1,-2)/math.sqrt(128)-scale[:,None,None]).exp();g=net.log_matrix(q,k).exp()
    z=t.mean(-1);gg=g.mean(-1);optimal=g*(z/gg)[...,None]
    a=(torch.arange(37,device='cuda',dtype=torch.float64)/37-.5).exp()[None,:,None]/gg[...,None]
    div=lambda x,y:x*(x.log()-y.log())-x+y
    lhs=div(t,a*g).mean(-1);rhs=div(t,optimal).mean(-1)+div(z,(a*g).mean(-1))
    projection=float(((lhs-rhs).abs()/(1+lhs.abs())).max());assert projection<1e-12
    # The moment method used by the exhaustive empirical-product evaluator.
    fq=net.log_feature(q,'q').exp();fk=net.log_feature(k,'k').exp()
    gram_energy=((fq.transpose(-1,-2)@fq)*(fk.transpose(-1,-2)@fk)).sum((-1,-2))
    dense_energy=g.square().sum((-1,-2))
    energy_error=float(((gram_energy-dense_energy).abs()/dense_energy).max());assert energy_error<1e-12
    products=[]
    for ds,n in [('fresh_wiki3',65536),('swde3',10240)]:
        d=read(f'results/product_{ds}.json');assert d['nq']==n and d['nk']==n and d['pairs_per_head']==n*n
        assert len(d['results'])==7 and any(r['method']=='favor_plus' for r in d['results'])
        for r in d['results']:
            assert all(math.isfinite(x) and x>=-1e-10 for key in ['relative_squared_risk','relative_I_divergence','query_balanced_divergence','directional_kl','mass_divergence'] for x in r[key])
        products.append(dict(dataset=ds,pairs_per_head=n*n,methods=7))
    checks=read('checks/implementation.json');assert checks['disjoint_token_hashes']
    stages={f:read('results/'+f) for f in ['stages.json','stages2.json','stages3.json']}
    assert all(r['exit_code']==0 for rows in stages.values() for r in rows)
    hashes={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(P.glob('*.py'))}
    result=dict(passed=True,fits=38,primary_single_pass_fits=29,calibrated_readout_fits=3,closed_form_global_calibrations=6,
        fitted_network_parameter_budgets_verified=True,ppl_cases=len(ppl),ppl_token_weighting_verified=True,no_invalid_ppl_outputs=True,
        ppl_round3_gauge_changes=deltas,projection_relative_error=projection,moment_energy_relative_error=energy_error,
        full_empirical_products=products,split_counts=checks['split_counts'],swde_source_files_disjoint=True,
        sources_sha256=hashes,scope='Only four frozen-model heads; hash disjointness does not prove absence of all semantic duplicates or population convergence.')
    save(P/'checks/final_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='sources_sha256'}),flush=True)

if __name__=='__main__':main()
