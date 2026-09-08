"""Exactly parameter-matched two-layer MLP, KAN and two-stage multiplicative KAN."""
import math
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'real_llm_pilot'))
from fit_features import HeadSpline


class Linear(nn.Module):
    def __init__(self,h,d,o):
        super().__init__();self.weight=nn.Parameter(torch.randn(h,o,d)/math.sqrt(d))
    def forward(self,x):return x@self.weight.transpose(-1,-2)


class Spline(HeadSpline):
    def __init__(self,h,d,o):
        super().__init__(h,d,o,grid=8,degree=3)
        self.base.register_parameter('bias',None)
    def forward(self,x):
        return F.silu(x)@self.base.weight.transpose(-1,-2)+self.basis(x).flatten(-2)@self.coefficients.transpose(-1,-2)


def multiply_nodes(x,additions,multiplications):
    if not multiplications:return x
    products=x[...,additions:].reshape(*x.shape[:-1],multiplications,2).prod(-1)
    return torch.cat([x[...,:additions],products],-1)


class Network(nn.Module):
    def __init__(self,h,method,scale):
        super().__init__();self.method=method;self.scale=scale
        if method=='mlp':
            self.first=Linear(h,128,192*scale);self.second=Linear(h,192*scale,64)
        elif method=='kan':
            self.first=Spline(h,128,16*scale);self.second=Spline(h,16*scale,64)
        elif method=='mulkan':
            self.first=Spline(h,128,15*scale);self.second=Spline(h,12*scale,96)
        else:raise ValueError(method)
        self.output_bias=nn.Parameter(torch.zeros(h,64))
        with torch.no_grad():
            if isinstance(self.second,Linear):self.second.weight.mul_(.1)
            else:
                self.second.base.weight.mul_(.1);self.second.coefficients.mul_(.1)
    def forward(self,x):
        x=self.first(x)
        if self.method=='mlp':x=F.silu(x)
        elif self.method=='mulkan':x=multiply_nodes(x,9*self.scale,3*self.scale)
        x=self.second(x)
        if self.method=='mulkan':x=multiply_nodes(x,32,32)
        return x+self.output_bias[:,None]


class FeaturePair(nn.Module):
    def __init__(self,method,normalization,scale=1):
        super().__init__();h=normalization['q_mean'].shape[0]
        self.method=method;self.scale=scale;self.heads=h;self.m=64
        for key,value in normalization.items():self.register_buffer(key,value)
        self.qnet=Network(h,method,scale);self.knet=Network(h,method,scale)
    def log_feature(self,x,side):
        x=(x-getattr(self,side+'_mean')[:,None])/getattr(self,side+'_std')[:,None]
        z=getattr(self,side+'net')(x)
        threshold=-40 if z.dtype==torch.float64 else -20
        # Stable log(softplus(z)); asymptotic branch is below the dtype's precision.
        log_sp=torch.where(z<threshold,z,F.softplus(z.clamp_min(threshold)).log())
        return log_sp-math.log(math.log(2))-.5*math.log(self.m)
    def log_kernel_pairs(self,q,k):return (self.log_feature(q,'q')+self.log_feature(k,'k')).logsumexp(-1)


def loss_per_pair(log_prediction,log_target,kind):
    if kind=='logcosh':
        r=(log_prediction-log_target).abs()
        return r+F.softplus(-2*r)-math.log(2)
    if kind=='poisson':
        # Constants depending only on the target are omitted from the training loss.
        return log_prediction.exp()-log_target.exp()*log_prediction
    raise ValueError(kind)


def verify():
    torch.manual_seed(123)
    norm={f'{side}_{stat}':(torch.zeros(1,128) if stat=='mean' else torch.ones(1,128))
          for side in ['q','k'] for stat in ['mean','std']}
    counts={}
    for scale in [1,2]:
        for method in ['mlp','kan','mulkan']:
            model=FeaturePair(method,norm,scale)
            count=sum(p.numel() for p in model.parameters());counts[f'{method}_{scale}']=count
            assert count==2*(36864*scale+64)
            q=torch.randn(1,9,128);k=torch.randn_like(q)
            logp=model.log_kernel_pairs(q,k)
            for loss in ['logcosh','poisson']:
                model.zero_grad();value=loss_per_pair(model.log_kernel_pairs(q,k),torch.randn(1,9),loss).mean()
                value.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            assert torch.isfinite(logp).all()
    x=torch.randn(2,3,15,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda z:multiply_nodes(z,9,3),(x,))
    logp=torch.tensor([-.7,0.,1.2],dtype=torch.float64,requires_grad=True)
    for loss in ['logcosh','poisson']:
        d=torch.autograd.grad(loss_per_pair(logp,logp.detach(),loss).sum(),logp,retain_graph=True)[0]
        assert d.abs().max()<1e-12
    print({'parameter_counts':counts,'multiplication_gradcheck':True,'loss_stationarity_at_exact_kernel':True})


if __name__=='__main__':verify()
