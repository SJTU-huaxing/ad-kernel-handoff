"""Matrix-free empirical spectral certificates, never deployment fitting.

For any orthonormal Q, s(Q^T K) are lower bounds on s(K).
Thus tail(s^2) <= optimal rank-m error <= ||K||_F^2-top_m(s^2).
Randomness selects the subspace; validity of this deterministic bracket does
not rely on a randomized-SVD success probability. Population inference still
requires sampling/tail assumptions absent here.
"""
import argparse,json,math,time
import torch
from runtime import P,OP
from run import load_data,HEADS

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rank',type=int,default=256)
    args=parser.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    data=load_data();scale=torch.load(OP/'results/train_basis.pt',weights_only=True)['scale'].cuda()
    path=P/'results/tightened_spectrum.json';rows=json.loads(path.read_text())['results'] if path.exists() else []
    for dataset in ['official','internal']:
        q,k=data[dataset]['q'],data[dataset]['k'];n=q.shape[1]
        package=torch.load(OP/f'results/{dataset}_moments.pt',weights_only=True)
        energy=package['accumulators'][str(n)]['energy']
        positive=json.loads((OP/f'results/{dataset}_summary.json').read_text())[-1]['heads']
        for h,head in enumerate(HEADS):
            if any(r['dataset']==dataset and r['head']==head and r['subspace_rank']==args.rank and r['power_iterations']==1 for r in rows):continue
            x=q[h].double();y=k[h].double();started=time.perf_counter();passno=0
            def apply(z,transpose=False,check_energy=False):
                nonlocal passno
                left,right=(y,x) if transpose else (x,y)
                out=torch.empty((n,z.shape[1]),device='cuda',dtype=torch.float64);h2=torch.zeros((),device='cuda',dtype=torch.float64)
                for begin in range(0,n,2048):
                    kernel=left[begin:begin+2048]@right.T/math.sqrt(128)
                    kernel.sub_(scale[h]).exp_()
                    out[begin:begin+2048]=kernel@z
                    if check_energy:h2+=kernel.square().sum()
                passno+=1
                if check_energy:
                    discrepancy=abs(float(h2)-float(energy[h]))/float(energy[h]);assert discrepancy<1e-9
                print(json.dumps(dict(event='spectral_pass',dataset=dataset,head=head,passno=passno,seconds=time.perf_counter()-started)),flush=True)
                return out
            gen=torch.Generator(device='cuda').manual_seed(20260920+h)
            omega=torch.randn((n,args.rank),device='cuda',dtype=torch.float64,generator=gen)
            basis=torch.linalg.qr(apply(omega,check_energy=True),mode='reduced').Q
            del omega
            for power in [0,1]:
                ortho=float((basis.T@basis-torch.eye(args.rank,device='cuda')).abs().max());assert ortho<1e-8
                projected=apply(basis,transpose=True)
                singular=torch.linalg.svdvals(torch.linalg.qr(projected,mode='r').R)
                h2=float(energy[h]);missed=max(0,1-float(singular.square().sum())/h2)
                bounds={}
                for m in [16,32,64,128]:
                    lower=float(singular[m:].square().sum())/h2;upper=max(0,1-float(singular[:m].square().sum())/h2)
                    risk=positive[h]['positive_frozen'][str(m)]['nmse'] if 'positive_frozen' in positive[h] else positive[h]['positive'][str(m)]['nmse']
                    bounds[str(m)]=dict(lower=lower,upper=upper,frozen_positive_risk=risk,
                        certified_empirical_suboptimality_ratio_at_least=risk/upper if upper else None)
                row=dict(dataset=dataset,head=head,n=n,subspace_rank=args.rank,power_iterations=power,
                    orthogonality_error=ortho,missed_energy_fraction=missed,bounds=bounds,seconds=time.perf_counter()-started)
                rows.append(row)
                path.write_text(json.dumps(dict(results=rows,
                    protocol='All 131072 internal / 10240 official held-out marginal samples. Float64 streaming raw exp(q.k/sqrt(128)), fixed numerical head scale. Seed-fixed randomized range + one subspace iteration; no deployment parameters changed.',
                    theorem='For orthonormal Q, sum_{j>m}s_j(Q^T K)^2 <= E_m*(K) <= ||K||_F^2-sum_{j<=m}s_j(Q^T K)^2. Width = missed Frobenius energy. Numerical orthogonality and independent energy checked.',
                    limitation='Deterministic certificates for the complete finite empirical product operator only. These are NOT confidence bounds for the unknown population operator. Near-population-optimality is not established.'),indent=2,allow_nan=False))
                print(json.dumps(row),flush=True)
                if power==0:
                    keybasis=torch.linalg.qr(projected,mode='reduced').Q
                    basis=torch.linalg.qr(apply(keybasis),mode='reduced').Q
            del x,y,basis,projected

if __name__=='__main__':main()
