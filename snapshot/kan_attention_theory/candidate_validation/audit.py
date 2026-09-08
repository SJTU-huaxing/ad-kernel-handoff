"""CPU-only final completeness and old/new protocol consistency audit."""
import hashlib,json,math
from pathlib import Path
import torch

P=Path(__file__).resolve().parent;ROOT=P.parent;OP=ROOT/'distribution_operator';COMP=ROOT/'kernel_comparison'

def read(path):return json.loads(path.read_text())

def main():
    torch.set_num_threads(4)
    kernel=[]
    for path in (P/'results').glob('kernel_*.json'):kernel+=read(path)['results']
    unique={(r['dataset'],r['n'],tuple(r['head']),r['method']) for r in kernel};assert len(unique)==len(kernel)
    assert all(math.isfinite(r['kernel_nmse']) and r['kernel_nmse']>=0 for r in kernel)
    raw=read(P/'results/raw_basis_protocol.json');ppl=read(P/'results/ppl.json')['results']
    methods={'cone_mulkan','cone_mlp','vq',*[f'nn_{m}_{s}' for m in ['mlp','kan','mulkan'] for s in [11,29,47]],*[r['method'] for r in raw['methods']]}
    assert len(ppl)==2*len(methods),(len(ppl),methods)
    for ds,count in [('internal',256),('official',20)]:
        for name in methods:
            row=next(r for r in ppl if r['dataset']==ds and r['method']==name)
            assert row['documents_count']==count and len(row['documents'])==count
            nll=sum(r['nll']*r['tokens'] for r in row['documents'])/sum(r['tokens'] for r in row['documents'])
            assert abs(nll-row['nll'])<1e-12 and abs(math.exp(nll)-row['perplexity'])<1e-10
            assert row['nonpositive_denominators']==0 and row['nonfinite_attention_rows']==0
    scale=torch.load(OP/'results/train_basis.pt',weights_only=True)['scale']
    energies=[];errors=[]
    for ds,n in [('internal',131072),('official',10240)]:
        old=torch.load(OP/f'results/{ds}_moments.pt',weights_only=True)['accumulators'][str(n)]['energy']
        for h,head in enumerate([[14,0],[14,6],[27,0],[27,6]]):
            reference=float((old[h]/n**2).log()+2*scale[h])
            rs=[r for r in kernel if r['dataset']==ds and r['n']==n and r['head']==head]
            err=max(abs(r['log_squared_energy_mean']-reference) for r in rs);energies.append(err);assert err<1e-10,(ds,head,err)
            if ds=='official':
                oldrf=read(COMP/f'results/official_L{head[0]}H{head[1]}.json')['results']
                for r in rs:
                    if not r['method'].startswith(('favor_','aderf_')) or r['method']=='favor_640':continue
                    family,seed=r['method'].split('_');ref=next(v['nmse'] for v in oldrf if v['m']==64 and v['seed']==int(seed) and v['method']=={'favor':'favor_plus','aderf':'aderf'}[family])
                    err=abs(r['kernel_nmse']-ref)/max(1,ref);errors.append(err);assert err<1e-8
    bench=read(P/'results/model_benchmark.json')['results']
    assert len(bench)==36 and len({(r['method'],r['prompt_tokens']) for r in bench})==36,len(bench)
    assert all(r['kv_cache_bytes']==28*2**20*r['prompt_tokens']//1024 for r in bench)
    assert all(r['linear_state_bytes']==(0 if r['method']=='teacher' else 132096) for r in bench)
    feature=read(P/'results/feature_benchmark.json')['results'];assert len(feature)==19*4*2,len(feature)
    stages=read(P/'results/stages.json')+read(P/'results/raw_stages.json');assert all(r['returncode']==0 for r in stages)
    manifest=read(ROOT/'single_pass_mulkan/data/manifest.json')
    hashes={s:{r['token_sha256'] for r in manifest['documents'] if r['split']==s} for s in ['train','validation','test']}
    assert all(not hashes[a]&hashes[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
    checkpoints=[]
    for m in ['mlp','kan','mulkan']:
        for seed in [11,29,47]:
            f=ROOT/f'single_pass_mulkan/fits/poisson_{m}_n4096_b1_s{seed}.json';meta=read(f)
            assert meta['parameters_per_qk_pair']==73856 and meta['epochs']==1 and meta['loss']=='poisson'
            checkpoints.append(dict(method=m,seed=seed,parameters=meta['parameters_per_qk_pair']))
    out=dict(kernel_rows=len(kernel),ppl_rows=len(ppl),new_full_model_document_forwards=sum(r['documents_count'] for r in ppl),
        end_to_end_timing_rows=len(bench),feature_timing_rows=len(feature),max_reference_log_energy_error=max(energies),
        max_recomputed_old_rf_relative_error=max(errors),all_neural_parameter_budgets_equal=True,
        disjoint_document_token_hashes=True,all_stages_passed=True,checkpoints=checkpoints,
        source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(P.glob('*.py'))})
    (P/'checks/audit.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['source_sha256','checkpoints']}))

if __name__=='__main__':main()
