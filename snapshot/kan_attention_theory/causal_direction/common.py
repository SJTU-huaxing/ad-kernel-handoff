import json,math,sys,time,hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
P=Path(__file__).resolve().parent;ROOT=P.parent
sys.path.insert(0,str(ROOT/'mlp_direction'))
from core import Pair as OriginalPair
H=24;D=128;HEADS=[[l,h] for l in [14,27] for h in range(12)];KI=torch.tensor([i//6 for i in range(H)])
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def data(split,device='cpu'):
 rows=[torch.load(p,weights_only=True) for p in sorted((P/'data').glob(f'{split}_*.pt'))]
 return {k:torch.cat([r[k] for r in rows]).to(device) for k in rows[0]}
def norm_scale(ds):
 norms={}
 for side in ['q','k']:
  x=ds[side][:,:,::8].permute(1,0,2,3).flatten(1,2).double()
  mean=x.mean(1).float();std=x.std(1).clamp_min(.03).float()
  if side=='k':mean=mean[KI];std=std[KI]
  norms[side+'_mean']=mean;norms[side+'_std']=std
 logs=[]
 for i in range(0,len(ds['q']),32):
  q=ds['q'][i:i+32].float();k=ds['k'][i:i+32,:,::16][:,KI].float()
  logs.append((q*k).sum(-1)/math.sqrt(D))
 x=torch.cat(logs).permute(1,0,2).flatten(1).double()
 return norms,(x.logsumexp(-1)-math.log(x.shape[1])).float()

class Baseline(nn.Module):
 def __init__(self,norm,kind,m=64):
  super().__init__();self.kind=kind;self.m=m;self.variant=kind;self.heads=H
  for k,v in norm.items():self.register_buffer(k,v)
  if kind=='hedgehog':
   width=m//2;w=torch.zeros(H,width,D);w[:,:, :width]=torch.eye(width)[None]
   self.weight=nn.Parameter(w);self.bias=nn.Parameter(torch.zeros(H,width))
  else:
   g=torch.randn(H,D,D);u,r=torch.linalg.qr(g);u=u*r.diagonal(dim1=-2,dim2=-1).sign()[:,None]
   rad=torch.randn(H,D,D).norm(dim=-1)
   self.weight=nn.Parameter(u.transpose(-1,-2)[:,:m]*rad[:,:m,None]);self.bias=nn.Parameter(torch.zeros(H,m))
 def log_feature(self,x,side):
  z=x*D**(-.25);p=z@self.weight.transpose(-1,-2)
  if self.kind=='hedgehog':
   p=p+self.bias[:,None];return torch.cat([p,-p],-1)-.5*math.log(self.m)
  return p-.5*z.square().sum(-1,keepdim=True)+self.bias[:,None]/2-.5*math.log(self.m)
 log_matrix=OriginalPair.log_matrix

def make(norm,kind):
 if kind in ['hedgehog','learned_prf','favor']:return Baseline(norm,'learned_prf' if kind=='favor' else kind)
 variant='factorized_both' if kind.startswith('split') else 'gauge_calibrated' if kind in ['calibrated','gate'] else 'exp' if kind in ['exp_kl','global_bal','global_raw'] else 'softplus'
 return OriginalPair(norm,64,variant)

def load(name,dtype=torch.float64):
 directory=ROOT/'orbit_direction' if name.startswith('orbit_') else P
 box=torch.load(directory/'fits'/f'{name}.pt',weights_only=True);meta=box['metadata'];s=box['state_dict'];norm={k:s[k] for k in ['q_mean','q_std','k_mean','k_std']}
 net=make(norm,meta['kind']).to(dtype);net.load_state_dict(s);return net.cuda().eval(),meta

def rows_loss(lp,lt,mask,lam):
 # Mask only specifies which pairs are observed; raw-kernel amplitudes retained.
 t=lt.masked_fill(~mask,-torch.inf);p=lp.masked_fill(~mask,-torch.inf)
 z=t.logsumexp(-1);zh=p.logsumexp(-1);a=(t-z[...,None]).exp()
 residual=(lt-z[...,None]-lp+zh[...,None]).masked_fill(~mask,0)
 kl=(a*residual).sum(-1);r=zh-z;mass=torch.expm1(r)-r
 return (kl+lam*mass if lam else kl),kl,mass

@torch.inference_mode()
def evaluate(net,ds,scale):
 rows=[];index=KI.cuda();scale=scale.cuda().double()
 for i in range(len(ds['q'])):
  q=ds['q'][i].cuda().double();k=ds['k'][i].cuda().double()[index];v=ds['v'][i].cuda().double()[index]
  pos=ds['query_positions'][i].cuda();mask=(torch.arange(k.shape[1],device='cuda')[None,:]<=pos[:,None])[None]
  lt=q@k.transpose(-1,-2)/math.sqrt(D)-scale[:,None,None];lp=net.log_matrix(q,k)
  balanced,kl,mass=rows_loss(lp,lt,mask,1.)
  t=lt.masked_fill(~mask,-torch.inf);p=lp.masked_fill(~mask,-torch.inf)
  a=t.softmax(-1);b=p.softmax(-1);y=a@v;yh=b@v
  truth=t.exp();pred=p.exp()
  d=dict(ordinal=i,kl=kl.mean(-1).tolist(),mass=mass.mean(-1).tolist(),balanced=balanced.mean(-1).tolist(),
   raw_sse=(truth-pred).square().sum((-1,-2)).tolist(),raw_energy=truth.square().sum((-1,-2)).tolist(),
   output_nmse=((y-yh).square().sum((-1,-2))/y.square().sum((-1,-2)).clamp_min(1e-30)).tolist())
  assert all(math.isfinite(x) for key in d if key!='ordinal' for x in d[key]);rows.append(d)
 summ={k:torch.tensor([r[k] for r in rows],dtype=torch.float64).mean(0).tolist() for k in rows[0] if k!='ordinal'}
 summ['raw_nmse']=(torch.tensor(summ['raw_sse'],dtype=torch.float64)/torch.tensor(summ['raw_energy'],dtype=torch.float64)).tolist()
 return dict(summary=summ,documents=rows)
