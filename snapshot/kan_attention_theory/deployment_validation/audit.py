"""Independent small exact-SVD checks plus complete-result consistency checks."""
import hashlib,json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent

def main():
    rng=np.random.default_rng(20260921);synthetic=[]
    for case in range(3):
        q=rng.normal(size=(128,8));k=rng.normal(size=(128,8))
        if case==1:q[:4]*=3
        if case==2:k+=2*(rng.random((128,1))<.05)
        logits=q@k.T/np.sqrt(8);kernel=np.exp(logits-logits.max());energy=np.sum(kernel**2)
        s=np.linalg.svd(kernel,compute_uv=False)
        basis=np.linalg.qr(kernel@rng.normal(size=(128,32)))[0]
        c=basis.T@kernel;u,t,vh=np.linalg.svd(c,full_matrices=False)
        for m in [4,8,16]:
            lower=np.sum(t[m:]**2)/energy;upper=1-np.sum(t[:m]**2)/energy
            exact=np.sum(s[m:]**2)/energy
            fit=basis@(u[:,:m]*t[:m])@vh[:m]
            feasible=np.sum((kernel-fit)**2)/energy
            assert lower-1e-10<=exact<=upper+1e-10
            assert abs(feasible-upper)<1e-10
            synthetic.append(dict(case=case,m=m,lower=lower,exact=exact,upper=upper,
                feasible_identity_error=abs(feasible-upper)))
    results=P/'results';spectrum=json.loads((results/'tightened_spectrum.json').read_text())['results']
    assert len(spectrum)==16
    for r in spectrum:
        for b in r['bounds'].values():
            assert b['lower']<=b['upper']+1e-10
            assert b['frozen_positive_risk']>=b['lower']-1e-10
            assert abs((b['upper']-b['lower'])-r['missed_energy_fraction'])<1e-10
        if r['power_iterations']==1:
            old=next(x for x in spectrum if all(x[k]==r[k] for k in ['head','dataset']) and x['power_iterations']==0)
            assert r['missed_energy_fraction']<=old['missed_energy_fraction']+1e-10
    ppl=json.loads((results/'ppl_optimized.json').read_text())['results'];assert len(ppl)==20
    assert sum(r['documents_count'] for r in ppl)==2760
    assert all(r['invalid_documents']==0 for r in ppl)
    assert all(r['nonpositive_denominators']==0 for r in ppl if r['method'] not in ['galerkin'])
    bench=json.loads((results/'model_benchmark_optimized.json').read_text())['results'];assert len(bench)==15
    for n in [1024,4096,8192]:assert len({r['kv_cache_bytes'] for r in bench if r['prompt_tokens']==n})==1
    hashes={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(P.glob('*.py'))}
    (results/'audit.json').write_text(json.dumps(dict(status='passed',synthetic_exact_svd_checks=synthetic,
        empirical_spectral_brackets=64,ppl_scenarios=20,document_forwards=2760,
        benchmark_rows=15,source_sha256=hashes,
        limitation='Numeric consistency does not certify unknown-population inference or production-engine optimality.'),indent=2))
    print('Audit passed.')

if __name__=='__main__':main()
