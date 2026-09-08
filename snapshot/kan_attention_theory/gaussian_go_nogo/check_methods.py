"""Numerical checks for NMF, nonzero-mean spectra, and the faithful FAVOR+ kernel."""
import json
import math
from pathlib import Path
import torch
from run import orf,log_sse,nmf_start,optimize_nmf

P=Path(__file__).resolve().parent/'results'


@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.manual_seed(819)
    # These small GPU checks may be run after the experiment to avoid resource overlap.
    q=torch.randn(2,13,128,device='cuda',dtype=torch.float64)*.2
    k=torch.randn(2,17,128,device='cuda',dtype=torch.float64)*.2
    omega=orf(2,819);m=32;beta=128**(-.5)
    lq=q@omega[:,:m].transpose(-1,-2)*128**(-.25)-q.square().sum(-1,keepdim=True)*beta/2
    lk=k@omega[:,:m].transpose(-1,-2)*128**(-.25)-k.square().sum(-1,keepdim=True)*beta/2
    direct=lq.exp()@lk.exp().transpose(-1,-2)/m
    lp=(lq[:,:,None]+lk[:,None]).logsumexp(-1)-math.log(m)
    error=float((direct-lp.exp()).norm()/direct.norm());assert error<1e-12
    target=q@k.transpose(-1,-2)*beta
    a=(direct-target.exp()).square().sum((-2,-1))
    b=log_sse(lp,target).exp();assert float((a-b).norm()/a.norm())<1e-12
    # The exact common-mean correction reconstructs q.k algebraically.
    muq=torch.randn(2,128,device='cuda',dtype=torch.float64);muk=torch.randn_like(muq)
    qc=q-muq[:,None];kc=k-muk[:,None]
    correction=(qc*muk[:,None]).sum(-1)[:,:,None]+(kc*muq[:,None]).sum(-1)[:,None,:]+(muq*muk).sum(-1)[:,None,None]
    ce=float(((qc@kc.transpose(-1,-2)+correction)*beta-target).abs().max());assert ce<1e-12
    # Known exact factors give a fixed point; cold starts must lower the objective.
    W=torch.rand(2,24,3,device='cuda',dtype=torch.float64);H=torch.rand(2,3,24,device='cuda',dtype=torch.float64)
    A=W@H;scale=A.norm(dim=(-2,-1)).sqrt();A/=scale[:,None,None].square()
    exactW=W/scale[:,None,None];exactH=H/scale[:,None,None]
    ew,eh,ee,_=optimize_nmf(A,exactW,exactH,20)
    assert (ee<1e-25).all()
    svd=torch.linalg.svd(A);wi,hi=nmf_start(A,svd,3,'nndsvd',881)
    initial=(A-wi@hi).square().sum((-2,-1))
    w,h,e,history=optimize_nmf(A,wi,hi,3000)
    assert (e<initial).all() and (w>=0).all() and (h>=0).all()
    assert all(all(b<=a+1e-12 for a,b in zip(p['error'],n['error'])) for p,n in zip(history,history[1:]))
    (P/'method_checks.json').write_text(json.dumps(dict(favor_log_vs_direct_relative_l2=error,
        mean_correction_max_absolute_error=ce,nmf_exact_rank3_fixed_point_errors=ee.tolist(),
        nmf_rank3_cold_start_errors=e.tolist(),nmf_rank3_initial_errors=initial.tolist(),
        nmf_nonnegative=True,nmf_monotone=True,
        note='NMF objective decrease and exact-factor fixed point verified; no global convergence guarantee.'),indent=2))


if __name__=='__main__':main()
