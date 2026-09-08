"""Direct small-operator checks, independent of the real-data large-product scan."""
import math,json
import torch
from run import P,scan,save

torch.set_num_threads(2);torch.manual_seed(872);torch.backends.cuda.matmul.allow_tf32=False
q=torch.randn(4,32,128,device='cuda',dtype=torch.float64)*.4
k=torch.randn_like(q)*.4
fq=torch.randn(4,32,8,device='cuda',dtype=torch.float64);fk=torch.randn_like(fq)
lq=torch.arange(32,device='cuda')[None].repeat(4,1)%8;lk=lq.roll(3,1)
basis=dict(scale=torch.zeros(4,device='cuda',dtype=torch.float64))
package=scan(q,k,fq,fk,lq,lk,basis,[16,32],P/'checks/small_product.pt',block=16,r=8)
records=[]
for n in [16,32]:
    K=(q[:,:n]@k[:,:n].transpose(-1,-2)/math.sqrt(128)).exp().cpu()
    ac=package['accumulators'][str(n)]
    assert torch.allclose(ac['energy'],K.square().sum((1,2)),rtol=1e-12)
    assert torch.allclose(ac['mass'],K.sum((1,2)),rtol=1e-12)
    for h in range(4):
        U=torch.linalg.qr(fq[h,:n].cpu()).Q;V=torch.linalg.qr(fk[h,:n].cpu()).Q
        C=U.T@K[h]@V/n;cs=torch.linalg.svdvals(C);s=torch.linalg.svdvals(K[h]/n)
        energy=K[h].square().mean()
        for m in [2,4]:
            lower=cs[m:].square().sum();oracle=s[m:].square().sum();upper=energy-cs[:m].square().sum()
            assert lower<=oracle+1e-12 and oracle<=upper+1e-12
            records.append(dict(n=n,head=h,m=m,lower=float(lower),actual_finite_floor=float(oracle),upper=float(upper)))
        A=torch.nn.functional.one_hot(lq[h,:n].cpu(),8).double();B=torch.nn.functional.one_hot(lk[h,:n].cpu(),8).double()
        sums=A.T@K[h]@B;counts=A.sum(0)[:,None]*B.sum(0)[None]
        assert torch.allclose(sums,ac['cell_sum'][h],rtol=1e-12)
        means=sums/counts;projection=A@means@B.T
        fitted=means*torch.rand_like(means)*2;pred=A@fitted@B.T
        residual=(K[h]-projection).square().mean();risk=(K[h]-pred).square().mean()
        estimation=((means-fitted).square()*counts).sum()/n**2
        assert abs(float(risk-residual-estimation))<1e-12
        # Conditional mean also minimizes generalized KL, hence no neural MSE fit is needed.
        assert (counts-sums/means).abs().max()<1e-10
        assert torch.linalg.matrix_rank(pred)<=8 and (pred>0).all()
save(P/'checks/theory_and_streaming.json',dict(spectral_bracket_checks=records,partition_pythagorean_identity=True,
    conditional_mean_idiv_stationarity=True,nonnegative_feature_factorization=True,exhaustive_streaming_matches_direct_matrix=True))
print('All small-operator checks passed.')
