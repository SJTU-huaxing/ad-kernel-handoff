"""Frozen positive random features: FAVOR+, centered FAVOR+, SDERF, ADERF.

Raw target exp(q.k/sqrt(d)); all statistics use training documents only.
SDERF/ADERF follow the NeurIPS 2023 DERF formulas, with Gaussian-marginal
orthogonal nodes. These are feature baselines, not complete FAVOR# systems.
"""
import json, math, sys
from pathlib import Path
import torch

P=Path(__file__).resolve().parent
OP=P.parent/'distribution_operator'
sys.path.insert(0,str(OP))
from run import load_data, save, HEADS, RANKS
D=128
SEEDS=[1009+37*i for i in range(5)]
METHODS=['favor_plus','centered_favor_plus','sderf','aderf']

def root(a,p):
    s,u=torch.linalg.eigh(a)
    assert s.min()>0
    return (u*s.pow(p)[:,None])@u.transpose(-1,-2)

def a_opt(lam):
    return (1-2*lam-torch.sqrt((2*lam+1).square()+8*lam))/16

def nodes(seed):
    gen=torch.Generator(device='cuda').manual_seed(seed)
    g=torch.randn(4,D,D,device='cuda',dtype=torch.float64,generator=gen)
    q,r=torch.linalg.qr(g);q=q*r.diagonal(dim1=-2,dim2=-1).sign()[:,None]
    rad=torch.randn(4,D,D,device='cuda',dtype=torch.float64,generator=gen).norm(dim=-1)
    return q.transpose(-1,-2)*rad[:,:,None]

@torch.no_grad()
def build_models(data):
    moments={};means={}
    for side in ['q','k']:
        n=0;first=torch.zeros(4,D,device='cuda',dtype=torch.float64)
        second=torch.zeros(4,D,D,device='cuda',dtype=torch.float64)
        for ds in ['calibration','moment_train']:
            xx=data[ds][side]
            for b in range(0,xx.shape[1],4096):
                x=xx[:,b:b+4096].double()*D**(-.25)
                first+=x.sum(1);second+=x.transpose(-1,-2)@x;n+=x.shape[1]
        means[side]=first/n;moments[side]=second/n
    mq,mk=means['q'],means['k']
    symmetric=moments['q']+moments['k']+mq[:,:,None]*mk[:,None]+mk[:,:,None]*mq[:,None]
    lam,rot=torch.linalg.eigh(symmetric);lam=lam.clamp_min(0)
    sde_a=a_opt(lam)
    rq,rk=root(moments['q'],.5),root(moments['k'],.5)
    u,s,vh=torch.linalg.svd(rq@rk)
    tq=root(moments['q'],-.5)@u*s.sqrt()[:,None]
    tk=root(moments['k'],-.5)@vh.transpose(-1,-2)*s.sqrt()[:,None]
    identity=torch.eye(D,device='cuda',dtype=torch.float64)[None]
    inv_error=(tq@tk.transpose(-1,-2)-identity).abs().max().item()
    assert inv_error<1e-7
    ade_a=a_opt((2*s.sum(-1)+2*(mq*mk).sum(-1))/D)
    models=[]
    for method in METHODS:
        for seed in SEEDS:
            omega=nodes(seed)
            row=dict(method=method,seed=seed,max_m=128,tq=None,tk=None)
            bias=torch.full((4,128),-.5*math.log(128),device='cuda',dtype=torch.float64)
            if method=='favor_plus':
                row.update(wq=omega.transpose(-1,-2),wk=omega.transpose(-1,-2),bq=bias,bk=bias)
            elif method=='centered_favor_plus':
                w=omega.transpose(-1,-2)+(mq+mk)[:,:,None]
                bq=bias-(omega*mq[:,None]).sum(-1)-.5*mq.square().sum(-1)[:,None]-.5*(mq*mk).sum(-1)[:,None]
                bk=bias-(omega*mk[:,None]).sum(-1)-.5*mk.square().sum(-1)[:,None]-.5*(mq*mk).sum(-1)[:,None]
                row.update(wq=w,wk=w,bq=bq,bk=bk)
            elif method=='sderf':
                bias=bias+(omega.square()*sde_a[:,None]).sum(-1)+.25*torch.log1p(-4*sde_a).sum(-1)[:,None]
                w=rot@(omega*torch.sqrt(1-4*sde_a)[:,None]).transpose(-1,-2)
                row.update(wq=w,wk=w,bq=bias,bk=bias)
            else:
                bias=bias+omega.square().sum(-1)*ade_a[:,None]+D/4*torch.log1p(-4*ade_a)[:,None]
                w=(omega*torch.sqrt(1-4*ade_a)[:,None,None]).transpose(-1,-2)
                row.update(wq=tq@w,wk=tk@w,bq=bias,bk=bias,tq=tq,tk=tk)
            models.append({k:v.cpu() if torch.is_tensor(v) else v for k,v in row.items()})
    torch.save(models,P/'results/rf_models.pt')
    torch.save(dict(means={k:v.cpu() for k,v in means.items()},moments={k:v.cpu() for k,v in moments.items()}),P/'results/rf_training_statistics.pt')
    save(P/'results/protocol.json',dict(methods=METHODS,seeds=SEEDS,ranks=RANKS,heads=HEADS,
        training_documents=4096,training_vectors_per_marginal=n,aderf_inverse_error=inv_error,
        statistics='Both disjoint training pools from distribution_operator; no evaluation statistics fit parameters.',
        target='exp(q^T k / sqrt(128)), no row normalization or clipping',
        nodes='Gaussian marginals via Haar orthogonal directions and independent chi_128 radii; nested prefixes across m.',
        comparison='Same frozen data and full empirical product as distribution_operator; five RF seeds per method.',
        implementation='SDERF/ADERF analytic feature formulas, not complete FAVOR# LLM reproductions.'))
    return models

class RandomFeatures:
    def __init__(self,model,m=64,dtype=torch.float64,device='cuda'):
        self.method=model['method'];self.m=m
        for side in ['q','k']:
            setattr(self,'w'+side,model['w'+side][:,:,:m].to(device=device,dtype=dtype))
            setattr(self,'b'+side,(model['b'+side][:,:m]+.5*math.log(model['max_m']/m)).to(device=device,dtype=dtype))
            t=model['t'+side]
            setattr(self,'t'+side,None if t is None else t.to(device=device,dtype=dtype))
        if torch.equal(model['wq'],model['wk']):self.wk=self.wq
        if torch.equal(model['bq'],model['bk']):self.bk=self.bq

    def log_features(self,x,side):
        x=x.to(getattr(self,'w'+side).dtype)*D**(-.25)
        t=getattr(self,'t'+side);z=x if t is None else x@t
        return x@getattr(self,'w'+side)+getattr(self,'b'+side)[:,None]-.5*z.square().sum(-1)[:,:,None]

    def features(self,x,side):return self.log_features(x,side).exp()

@torch.no_grad()
def bank_features_one_head(x,side,models,h,block=8192):
    """Store normalized Float64 features; returned log scales restore raw values."""
    w=torch.cat([r['w'+side][h] for r in models],-1).cuda()
    bias=torch.stack([r['b'+side][h] for r in models]).cuda()
    transforms={r['method']:r['t'+side][h].cuda() if r['t'+side] is not None else None for r in models}
    m=models[0]['max_m'];assert all(r['max_m']==m for r in models)
    n=len(x);f=torch.empty(n,len(models)*m,device='cuda',dtype=torch.float64)
    for b in range(0,n,block):
        xx=x[b:b+block].double()*D**(-.25)
        norm={method:(xx if t is None else xx@t).square().sum(-1)/2 for method,t in transforms.items()}
        base=torch.stack([norm[r['method']] for r in models],-1)
        f[b:b+block]=(xx@w).view(-1,len(models),m).add(bias[None]).sub(base[:,:,None]).flatten(1)
    scales=f.view(n,len(models),m).amax((0,2))
    for b in range(0,n,block):
        f[b:b+block].view(-1,len(models),m).sub_(scales[None,:,None]).exp_()
    assert torch.isfinite(f).all()
    return f,scales

def pooled_model(models,method):
    """Predefined union of all five orthogonal blocks, not best-seed selection."""
    rows=[r for r in models if r['method']==method];count=len(rows)
    result=dict(method=method,seed='pooled_5_blocks',max_m=128*count)
    for side in ['q','k']:
        result['w'+side]=torch.cat([r['w'+side] for r in rows],-1)
        result['b'+side]=torch.cat([r['b'+side] for r in rows],-1)-.5*math.log(count)
        result['t'+side]=rows[0]['t'+side]
    return result
