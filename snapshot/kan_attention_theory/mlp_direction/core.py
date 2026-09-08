"""Raw exponential-kernel MLPs: fixed hidden width, variable feature rank.

All learned primary objectives retain the unnormalized target. No LLM weights
are trained. Data and normalization are fixed before validation/test evaluation.
"""
import json, math, sys, hashlib, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

P=Path(__file__).resolve().parent
ROOT=P.parent
DATA=ROOT/'single_pass_mulkan/data'
HEADS=[[14,0],[14,6],[27,0],[27,6]]
for folder in ['fits','results','checks','figures']:(P/folder).mkdir(parents=True,exist_ok=True)

def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False))

class Network(nn.Module):
    def __init__(self,h,m):
        super().__init__()
        self.w1=nn.Parameter(torch.randn(h,192,128)/math.sqrt(128))
        self.w2=nn.Parameter(torch.randn(h,m,192)/math.sqrt(192)*.1)
        self.bias=nn.Parameter(torch.zeros(h,m))
    def forward(self,x):
        return F.silu(x@self.w1.transpose(-1,-2))@self.w2.transpose(-1,-2)+self.bias[:,None]

class Pair(nn.Module):
    def __init__(self,norm,m=64,variant='softplus'):
        super().__init__();self.m=m;self.heads=norm['q_mean'].shape[0];self.variant=variant
        for k,v in norm.items():self.register_buffer(k,v)
        self.qnet=Network(self.heads,m);self.knet=Network(self.heads,m)
    def log_feature(self,x,side):
        z=getattr(self,side+'net')((x-getattr(self,side+'_mean')[:,None])/getattr(self,side+'_std')[:,None])
        if self.variant=='exp' or (self.variant=='gauge_calibrated' and side=='k'):return z-.5*math.log(self.m)
        if self.variant=='factorized_both' or (self.variant in ['factorized','gauge_calibrated'] and side=='q'):
            # m-1 free directional logits + one amplitude coordinate: still m
            # outputs and exactly the original active parameter budget.
            shape=torch.cat([z[...,:-1],torch.zeros_like(z[...,-1:])],-1)
            return shape.log_softmax(-1)+z[...,-1:]+.5*math.log(self.m)
        threshold=-40 if z.dtype==torch.float64 else -20
        return torch.where(z<threshold,z,F.softplus(z.clamp_min(threshold)).log())-math.log(math.log(2))-.5*math.log(self.m)
    def log_matrix(self,q,k):
        a,b=self.log_feature(q,'q'),self.log_feature(k,'k')
        aq=a.amax(-1,keepdim=True);bk=b.amax(-1,keepdim=True)
        product=(a-aq).exp()@(b-bk).exp().transpose(-1,-2)
        return product.clamp_min(torch.finfo(product.dtype).tiny).log()+aq+bk.transpose(-1,-2)

def loss_rows(logp,logt,kind):
    """Each result has shape H,Q. All targets remain log(raw kernel / e^s_h)."""
    logz=logt.logsumexp(-1)
    logzh=logp.logsumexp(-1)
    probability=(logt-logz[...,None]).exp()
    r=logzh-logz
    directional=(probability*(logt-logz[...,None]-logp+logzh[...,None])).sum(-1)
    mass=torch.expm1(r)-r
    balanced=directional+mass
    if kind in ['kl_control','exp_control']:return directional
    if kind in ['balanced','factorized','factorized_both']:return balanced
    if kind=='raw':return balanced*(logz-math.log(logt.shape[-1])).exp()
    if kind=='half':return balanced*((logz-math.log(logt.shape[-1]))*.5).exp()
    if kind=='logcosh':
        a=(logp-logt).abs()
        return (a+F.softplus(-2*a)-math.log(2)).mean(-1)
    raise ValueError(kind)

@torch.no_grad()
def load_train():
    manifest=json.loads((DATA/'manifest.json').read_text())
    parts=[torch.load(DATA/r['file'],weights_only=True) for r in manifest['shards'] if r['split']=='train']
    data={s:torch.cat([b[s] for b in parts]).cuda() for s in ['q','k']}
    norm={}
    for s in ['q','k']:
        x=data[s][:,:,::32].permute(1,0,2,3).flatten(1,2).double()
        norm[s+'_mean']=x.mean(1).float();norm[s+'_std']=x.std(1).clamp_min(.03).float()
    # The same fixed head scale as previous one-pass fitting, calibrated on
    # all training paired logits, not a per-query renormalization.
    logits=[]
    for start in range(0,len(data['q']),64):
        logits.append((data['q'][start:start+64].float()*data['k'][start:start+64].float()).sum(-1)/math.sqrt(128))
    logs=torch.cat(logits).permute(1,0,2).flatten(1).double()
    scale=(logs.logsumexp(-1)-math.log(logs.shape[-1])).float()
    return manifest,data,norm,scale

def load_full(split):
    manifest=json.loads((DATA/'manifest.json').read_text())
    if split!='official':
        rows=[torch.load(DATA/r['file'],weights_only=True) for r in manifest['shards'] if r['split']==split]
        return {s:torch.cat([x[s] for x in rows]) for s in ['q','k','v','input_ids']}
    old=ROOT/'real_llm_pilot/data_qwen25_1p5b';om=json.loads((old/'manifest.json').read_text())
    ix=[om['head_labels'].index(h) for h in HEADS]
    rows=[torch.load(old/r['file'],weights_only=True) for r in om['documents'] if r['split']=='test']
    return {s:torch.stack([x[s][ix] if s!='input_ids' else x[s] for x in rows]) for s in ['q','k','v','input_ids']}

@torch.no_grad()
def evaluate(model,data,scale,limit=None,query_step=8):
    """Unseen real context rectangles; no test fitting or selected rows by error."""
    rows=[];mid=data['q'].shape[2]//2
    for i in range(min(limit or len(data['q']),len(data['q']))):
        q=data['q'][i,:,mid::query_step].cuda().double()
        k=data['k'][i,:,:mid].cuda().double();v=data['v'][i,:,:mid].cuda().double()
        logt=q@k.transpose(-1,-2)/math.sqrt(128)-scale.double()[:,None,None]
        logp=model.log_matrix(q,k)
        lt=logt.logsumexp(-1);lp=logp.logsumexp(-1)
        a=(logt-lt[...,None]).exp();b=(logp-lp[...,None]).exp()
        r=lp-lt;kl=(a*(logt-lt[...,None]-logp+lp[...,None])).sum(-1)
        mass=torch.expm1(r)-r
        y=a@v;yh=b@v
        truth=logt.exp();pred=logp.exp()
        dist=(v.square().sum(-1)[:,None]+y.square().sum(-1)[:,:,None]-2*y@v.transpose(-1,-2)).clamp_min(0).sqrt()
        w=dist+.05*(a*dist).sum(-1,keepdim=True)+1e-6
        aw=(w*truth).sum(-1);bw=(w*pred).sum(-1)
        dw=(w*(pred-truth+truth*(logt-logp))).sum(-1).clamp_min(0)
        bound=2*(aw+bw)*dw/lp.exp().square()
        plain_bound=2*dist.amax(-1).square()*(kl+mass).clamp_min(0)
        observed=(y-yh).square().sum(-1)
        assert (observed<=bound+1e-8*(1+bound)).all(), 'Weighted output certificate failed'
        vals=dict(raw_sse=(truth-pred).square().sum((-1,-2)),raw_energy=truth.square().sum((-1,-2)),
            balanced=(kl+mass).mean(-1),directional_kl=kl.mean(-1),mass=mass.mean(-1),
            log_mae=(logt-logp).abs().mean((-1,-2)),
            attention_sse=(a-b).square().sum((-1,-2)),attention_energy=a.square().sum((-1,-2)),
            output_sse=(y-yh).square().sum((-1,-2)),output_energy=y.square().sum((-1,-2)),
            output_nmse=(y-yh).square().sum((-1,-2))/y.square().sum((-1,-2)),
            value_divergence=(dw/aw).mean(-1),value_bound=bound.mean(-1),plain_bound=plain_bound.mean(-1),
            bound_ratio=(bound/plain_bound.clamp_min(1e-30)).mean(-1),
            row_l1=(a-b).abs().sum(-1).mean(-1))
        rows.append(dict(ordinal=i,**{key:val.tolist() for key,val in vals.items()}))
    t={key:torch.tensor([r[key] for r in rows],dtype=torch.float64) for key in rows[0] if key!='ordinal'}
    summary={key:value.mean(0).tolist() for key,value in t.items()}
    for a,b,key in [('raw_sse','raw_energy','raw_nmse'),('attention_sse','attention_energy','attention_nmse'),('output_sse','output_energy','output_pooled_nmse')]:
        summary[key]=(t[a].sum(0)/t[b].sum(0)).tolist()
    return dict(summary=summary,documents=rows,query_step=query_step,queries_per_doc=len(range(mid,data['q'].shape[2],query_step)),keys_per_doc=mid)

def load_fit(name,dtype=torch.float64):
    box=torch.load(P/'fits'/f'{name}.pt',weights_only=True)
    state=box['state_dict'];norm={k:state[k] for k in ['q_mean','q_std','k_mean','k_std']}
    model=Pair(norm,box['metadata']['m'],box['metadata'].get('variant','softplus')).to(dtype);model.load_state_dict(state)
    return model.cuda().to(dtype).eval(),box['metadata']
