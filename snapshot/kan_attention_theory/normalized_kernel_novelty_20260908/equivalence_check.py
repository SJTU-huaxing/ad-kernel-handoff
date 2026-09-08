"""Numerical illustrations of exact algebraic identities; no performance claim."""
import json
import sys
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'query_amplitude_ablation'))
import ablation as a
torch=a.torch

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    net,_=a.load_fit('causal_kl_reduced_plain_s11_lr0.002',torch.float64,'cpu')
    net.runtime_raw=True
    ds=a.source.data('confirm_wiki')
    q=ds['q'][0].double();k=ds['k'][0].double()[a.KI]
    mask=torch.arange(k.shape[1])[None]<=ds['query_positions'][0,:,None]
    original=net.log_matrix(q,k).masked_fill(~mask[None],-torch.inf).softmax(-1)
    qx=(q-net.q_mean[:,None])/net.q_std[:,None]
    kx=(k-net.k_mean[:,None])/net.k_std[:,None]
    zq=net.qnet(qx)
    zq=torch.cat([zq,torch.zeros_like(zq[...,:1])],-1)
    zk_all=net.knet(kx)
    zk=torch.cat([zk_all[...,:63],torch.zeros_like(zk_all[...,:1])],-1)
    sk=zk_all[...,-1:]
    logkey=sk+zk-zk.logsumexp(-1,keepdim=True)
    unnormalized=zq.exp()@logkey.exp().transpose(-1,-2)
    weights=unnormalized.masked_fill(~mask[None],0)
    weights/=weights.sum(-1,keepdim=True)
    exp_equivalence=float((weights-original).abs().max())
    # Per-prefix mixture of m context-defined key distributions.
    phi=net.raw_log_feature(k,'k').exp()
    piq=net.raw_log_feature(q,'q').exp()
    masses=phi.cumsum(1)[:,ds['query_positions'][0]]
    alpha=piq*masses
    alpha/=alpha.sum(-1,keepdim=True)
    mixture=torch.zeros_like(original)
    for h in range(a.H):
        prototypes=phi[h,None]*mask[:,:,None]/masses[h,:,None]
        mixture[h]=torch.einsum('qr,qkr->qk',alpha[h],prototypes)
    mixture_error=float((mixture-original).abs().max())
    # General positive features -> amplitude-direction normalized representative.
    gen=torch.Generator().manual_seed(7365)
    fq=torch.randn(32,64,generator=gen,dtype=torch.float64).exp()
    fk=torch.randn(127,64,generator=gen,dtype=torch.float64).exp()
    first=fq@fk.T;first/=first.sum(-1,keepdim=True)
    pq=fq/fq.sum(-1,keepdim=True);pk=fk/fk.sum(-1,keepdim=True)
    second=pq@(fk.sum(-1,keepdim=True)*pk).T;second/=second.sum(-1,keepdim=True)
    general_error=float((first-second).abs().max())
    assert max(exp_equivalence,mixture_error,general_error)<1e-12
    result=dict(passed=True,real_ad_exponential_rewrite_max_error=exp_equivalence,
        real_ad_causal_mixture_max_error=mixture_error,general_positive_rewrite_max_error=general_error,
        scope='Illustrates exact algebra only; does not prove finite-network parameter budget equivalence or new approximation guarantees.')
    (P/'equivalence_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

if __name__=='__main__':main()
