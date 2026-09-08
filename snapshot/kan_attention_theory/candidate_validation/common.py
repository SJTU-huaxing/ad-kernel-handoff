"""Frozen candidate feature maps. All raw kernels use one fixed head scale."""
import json, math, sys
from pathlib import Path
import torch
import torch.nn.functional as F

P=Path(__file__).resolve().parent
ROOT=P.parent; OP=ROOT/'distribution_operator'; COMP=ROOT/'kernel_comparison'
DEP=ROOT/'deployment_validation'; SINGLE=ROOT/'single_pass_mulkan'
sys.path.insert(0,str(OP));sys.path.insert(0,str(COMP));sys.path.insert(0,str(SINGLE))
from run import load_data, HEADS
from kernels import GalerkinFeatures, PartitionFeatures
from rf import RandomFeatures, pooled_model
from models import FeaturePair

def save(path,x):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(x,indent=2,allow_nan=False))

def cpu(x):
    if torch.is_tensor(x):return x.detach().cpu()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu(v) for v in x]
    return x

def learned(method,seed=11,dtype=torch.float64):
    ck=torch.load(SINGLE/f'fits/poisson_{method}_n4096_b1_s{seed}.pt',weights_only=True)
    state=ck['state_dict'];norm={s:state[s] for s in ['q_mean','q_std','k_mean','k_std']}
    net=FeaturePair(method,norm,1);net.load_state_dict(state);net=net.cuda().to(dtype).eval()
    return net,ck['metadata']

class Candidate:
    def __init__(self,name,dtype=torch.float64,indices=(0,1,2,3)):
        self.name=name;self.method=name;self.dtype=dtype;self.indices=list(indices);self.m=64
        self.params=torch.load(P/'results/construction.pt',weights_only=True)
        self.scale=self.params['scale'].cuda().to(dtype)
        self.net=None;self.base=None;self.meta={}
        if name.startswith('nn_'):
            _,method,seed=name.split('_');self.net,self.meta=learned(method,int(seed),dtype)
            self.logrestore=(torch.tensor(self.meta['log_scale'],device='cuda',dtype=dtype)-self.scale)/2
        elif name in ['partition','galerkin']:
            self.base=(PartitionFeatures if name=='partition' else GalerkinFeatures)(64,dtype)
        elif name.startswith(('favor_','aderf_','centered_')):
            family,seed=name.split('_');method={'favor':'favor_plus','aderf':'aderf','centered':'centered_favor_plus'}[family]
            bank=torch.load(COMP/'results/rf_models.pt',weights_only=True)
            row=pooled_model(bank,method) if seed=='640' else next(r for r in bank if r['method']==method and r['seed']==int(seed))
            self.m=640 if seed=='640' else 64;self.base=RandomFeatures(row,self.m,dtype)
        self.t={k:v.cuda().to(dtype) for k,v in self.params.items() if torch.is_tensor(v)}
        if name.startswith(('avg_','cone_')):
            kind=name.split('_',1)[1]
            self.kind=kind
            if kind!='anchor':
                self.net,self.meta=learned(kind.removeprefix('raw_'),11,dtype)
                del self.net.qnet
            box=self.params['bases'][kind]
            self.basis={k:v.cuda().to(dtype) for k,v in box.items() if torch.is_tensor(v)}
        if name.startswith(('avg_','cone_')):keep=['moment_k']+(['positive_q'] if self.kind=='anchor' else [])
        elif name=='vq':keep=['vq_centers']
        elif name=='nystrom':keep=['anchor_q','anchor_k','nys_q','nys_k']
        elif name=='spectral_pair':keep=['anchor_q','anchor_k','nys_q','nys_k','envelope_q','envelope_k','pair_c0','pair_delta']
        else:keep=[]
        self.t={k:self.t[k] for k in keep}
        if self.indices!=[0,1,2,3]:
            def sliced(v):
                return v[self.indices].contiguous() if torch.is_tensor(v) and v.ndim and v.shape[0]==4 else v
            self.scale=sliced(self.scale);self.t={k:sliced(v) for k,v in self.t.items()}
            if hasattr(self,'basis'):self.basis={k:sliced(v) for k,v in self.basis.items()}
            if hasattr(self,'logrestore'):self.logrestore=sliced(self.logrestore)
            if self.base is not None:
                for k,v in list(vars(self.base).items()):setattr(self.base,k,sliced(v))
            if self.net is not None:
                for mod in self.net.modules():
                    for k,v in list(mod._parameters.items()):
                        if v is not None:mod._parameters[k]=torch.nn.Parameter(sliced(v),requires_grad=False)
                    for k,v in list(mod._buffers.items()):mod._buffers[k]=sliced(v)

    def bfeatures(self,x,kind=None):
        kind=kind or self.kind
        if kind=='anchor':
            return (x@self.t['positive_q'].transpose(-1,-2)/math.sqrt(128)).softmax(-1)
        z=self.net.log_feature(x,'k');return z.exp() if kind.startswith('raw_') else z.softmax(-1)

    def nystrom(self,x,side,r):
        anchors=self.t['anchor_k' if side=='q' else 'anchor_q']
        proj=self.t['nys_'+side][:,:,:r]
        return (x@anchors.transpose(-1,-2)/math.sqrt(128)-self.scale[:,None,None]).exp()@proj

    def features_all(self,x,side):
        x=x.to(self.dtype)
        if self.name.startswith('nn_'):
            return (self.net.log_feature(x,side)+self.logrestore[:,None,None]).exp()
        if self.name in ['partition','galerkin']:
            return self.base.features(x,side)/self.base.sqrt_scale[:,None,None]
        if self.name.startswith(('favor_','aderf_','centered_')):
            return (self.base.log_features(x,side)-self.scale[:,None,None]/2).exp()
        if self.name=='nystrom':return self.nystrom(x,side,64)
        if self.name=='spectral_pair':
            f=self.nystrom(x,side,32);a=self.t['envelope_'+side]
            u=f[:,:,:1];signed=f[:,:,1:]
            first=self.t['pair_c0'].sqrt()[:,None,None]*u
            plus=(a[:,None]*u+signed)/math.sqrt(2)
            minus=(a[:,None]*u-signed)/math.sqrt(2)
            return F.pad(torch.cat([first,plus,minus],-1),(0,1))
        if self.name=='vq':
            centers=self.t['vq_centers']
            if side=='q':return (x@centers.transpose(-1,-2)/math.sqrt(128)-self.scale[:,None,None]).exp()
            dist=x.square().sum(-1,keepdim=True)+centers.square().sum(-1)[:,None]-2*x@centers.transpose(-1,-2)
            return F.one_hot(dist.argmin(-1),64).to(self.dtype)
        if self.name.startswith(('avg_','cone_')):
            if side=='k':return self.bfeatures(x)
            h=(x@self.t['moment_k'].transpose(-1,-2)/math.sqrt(128)-self.scale[:,None,None]).exp()@self.basis['weights']
            if self.name.startswith('avg_'):return h/self.basis['p'][:,None]
            # Fixed 128 projected-gradient iterations: feasible numerical
            # approximation to the cone oracle, not claimed to be exact NNLS.
            diag=self.basis['diag'];hs=h/diag[:,None]
            init=self.basis.get('initial_scale',self.basis['p'])
            z=(h/init[:,None])*diag[:,None]
            step=self.basis['lipschitz'][:,None,None]
            y=z;alpha=1.
            for _ in range(128):
                new=(y-(y@self.basis['correlation']-hs)/step).clamp_min(0)
                anew=(1+math.sqrt(1+4*alpha*alpha))/2
                y=new+(alpha-1)/anew*(new-z);z=new;alpha=anew
            return z/diag[:,None]
        raise ValueError(self.name)

    def features(self,x,side):
        # Existing networks are head-batched. Select their head parameters once
        # in runtime adapter; generic validation always calls all four heads.
        return self.features_all(x,side)

    def prefill(self,q,k):return self.features(q,'q'),self.features(k,'k'),None
    def decode(self,q,k,state):return self.features(q,'q'),self.features(k,'k')

def names():
    return ['nystrom','spectral_pair','avg_anchor','cone_anchor',
            'avg_mlp','cone_mlp','avg_kan','cone_kan','avg_mulkan','cone_mulkan',
            'vq','partition','galerkin']+[f'nn_{method}_{seed}' for method in ['mlp','kan','mulkan'] for seed in [11,29,47]]+[
            f'{method}_{seed}' for method in ['favor','aderf'] for seed in [1009,1046,1083,1120,1157]]+['favor_640']
