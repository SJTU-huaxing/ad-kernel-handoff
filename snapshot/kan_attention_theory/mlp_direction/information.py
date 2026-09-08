"""Population-valid information lower bound, numerically evaluated on empirical measures.

Positive m-feature conditional kernels form mixtures of m product components;
forward KL >= max(I(Q;K)-log(m), 0). The inequality is classical latent-variable
information theory, not claimed as a new theorem. This does not require L2/HS.
"""
from core import *

def measures(logk):
    lp=logk-logk.logsumexp(-1,keepdim=True)
    p=lp.exp();logz=logk.logsumexp(-1)
    res={}
    for mode in ['uniform_query','kernel_mass_query']:
        lq=torch.full_like(logz,-math.log(logz.shape[-1])) if mode=='uniform_query' else logz.log_softmax(-1)
        marginal=(lq[...,None]+lp).logsumexp(-2)
        info=(lq.exp()*(p*(lp-marginal[:,None])).sum(-1)).sum(-1)
        res[mode]=dict(mutual_information_nats=info.tolist(),effective_components=info.exp().tolist(),
            lower_bounds={str(m):(info-math.log(m)).clamp_min(0).tolist() for m in [16,32,64,96,128]})
    return res

@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    gen=torch.Generator().manual_seed(74601);manifest=json.loads((DATA/'manifest.json').read_text())
    q=[];k=[]
    for r in manifest['shards']:
        if r['split']!='train':continue
        b=torch.load(DATA/r['file'],weights_only=True)
        for i in range(len(b['q'])):
            ix=torch.randint(512,(2,),generator=gen)
            q.append(b['q'][i,:,ix[0]]);k.append(b['k'][i,:,ix[1]])
    q=torch.stack(q,1).cuda().double();k=torch.stack(k,1).cuda().double()
    result=measures(q@k.transpose(-1,-2)/math.sqrt(128))
    # Tightness family: diagonal exponential kernel, m equal disjoint blocks.
    n=128;m=16;g=n//m;logk=torch.eye(n,dtype=torch.float64)*30
    t=logk.exp();same=(torch.arange(n)//g)[:,None]==(torch.arange(n)//g)[None,:]
    pred=torch.where(same,torch.tensor(1+(math.exp(30)-1)/g),torch.tensor(1.)).double()
    p=t/t.sum(-1,keepdim=True);b=pred/pred.sum(-1,keepdim=True)
    kl=(p*(p.log()-b.log())).sum(-1).mean()
    mi=(p*(p.log()+math.log(n))).sum(-1).mean()
    assert kl>=mi-math.log(m)-1e-10
    assert abs(float(kl)-(float(mi)-math.log(m)))<1e-6
    save(P/'results/information.json',dict(train_product=result,toy=dict(n=n,m=m,kl=float(kl),mi=float(mi),lower_bound=float(mi)-math.log(m)),
        caveat='Real values refer to the 4096x4096 empirical train product; the inequality is population-valid under finite KL, but empirical MI is not a population confidence lower bound. May be vacuous at m>=16.'))
    print(json.dumps(dict(event='information',results=result)),flush=True)

if __name__=='__main__':main()
