"""Frozen m=64 kernels, causal chunk scan and true recurrent decode.

Raw kernels are unchanged. Common head scales and query-row scalings cancel
only when evaluating the linear-attention numerator/denominator.
"""
import json, math, sys
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

P = Path(__file__).resolve().parent
COMP = P.parent / 'kernel_comparison'
sys.path.insert(0, str(COMP))
from kernels import PartitionFeatures, GalerkinFeatures
from rf import RandomFeatures, OP, pooled_model

@triton.jit
def _route(C, A, T, L, R, LEAF, OUT, N:tl.constexpr, E:tl.constexpr,
           NN:tl.constexpr, DEPTH:tl.constexpr, BLOCK:tl.constexpr):
    h=tl.program_id(0); t=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    node=tl.full((BLOCK,),0,tl.int32)
    for _ in range(DEPTH):
        left=tl.load(L+h*NN+node)
        active=left>=0
        axis=tl.load(A+h*NN+node)
        val=tl.load(C+(h*N+t)*E+tl.maximum(axis,0),t<N,other=0)
        threshold=tl.load(T+h*NN+node)
        right=tl.load(R+h*NN+node)
        node=tl.where(active,tl.where(val>threshold,right,left),node)
    leaf=tl.load(LEAF+h*NN+node)
    tl.store(OUT+h*N+t,leaf,t<N)

@triton.jit
def _exp_scale(X,S,Y,N:tl.constexpr,A:tl.constexpr,B:tl.constexpr):
    h=tl.program_id(0);i=tl.program_id(1)*B+tl.arange(0,B)
    v=tl.load(X+h*N*A+i,i<N*A,other=0)
    s=tl.load(S+h)
    tl.store(Y+h*N*A+i,tl.exp(v*0.08838834764831845-s),i<N*A)

@triton.jit
def _favor_exp(PROJ,X,BIAS,Y,N:tl.constexpr,M:tl.constexpr,D:tl.constexpr,
               BM:tl.constexpr,BD:tl.constexpr,LOG:tl.constexpr=False):
    h=tl.program_id(0);t=tl.program_id(1)
    d=tl.arange(0,BD);r=tl.arange(0,BM)
    x=tl.load(X+(h*N+t)*D+d,d<D,other=0)
    norm=tl.sum(x*x,0)*0.04419417382415922
    z=tl.load(PROJ+(h*N+t)*M+r,r<M,other=0)
    bias=tl.load(BIAS+h*M+r,r<M,other=0)
    result=z+bias-norm
    if not LOG:result=tl.exp(result)
    tl.store(Y+(h*N+t)*M+r,result,r<M)

@triton.jit
def _landmark_gemv(X,W,S,Y,A:tl.constexpr,B:tl.constexpr):
    h=tl.program_id(0);a=tl.program_id(1)*B+tl.arange(0,B);d=tl.arange(0,128)
    x=tl.load(X+h*128+d)
    w=tl.load(W+(h*A+a[:,None])*128+d[None,:],a[:,None]<A,other=0)
    z=tl.sum(w*x[None,:],1)*0.08838834764831845-tl.load(S+h)
    tl.store(Y+h*A+a,tl.exp(z),a<A)

@triton.jit
def _favor_gemv(X,W,BIAS,Y,B:tl.constexpr,LOG:tl.constexpr=False):
    h=tl.program_id(0);r=tl.program_id(1)*B+tl.arange(0,B);d=tl.arange(0,128)
    x=tl.load(X+h*128+d)
    w=tl.load(W+h*128*64+d[None,:]*64+r[:,None],r[:,None]<64,other=0)
    z=tl.sum(w*x[None,:],1)-tl.sum(x*x,0)*0.04419417382415922
    z+=tl.load(BIAS+h*64+r,r<64,other=0)
    if not LOG:z=tl.exp(z)
    tl.store(Y+h*64+r,z,r<64)

@triton.jit
def _step(Q,K,V,S,Z,O,DEN,PARTITION:tl.constexpr):
    h=tl.program_id(0);r=tl.arange(0,64);d=tl.arange(0,128)
    q=tl.load(Q+h*64+r)
    v=tl.load(V+h*128+d);s=tl.load(S+h*64*128+r[:,None]*128+d[None,:])
    z=tl.load(Z+h*64+r)
    if PARTITION:
        cell=tl.load(K+h);weight=tl.load(Q+h*64+cell)
        den=tl.sum(q*z,0)+weight
        out=(tl.sum(q[:,None]*s,0)+weight*v)/den
        bucket=tl.load(S+h*64*128+cell*128+d)
        count=tl.load(Z+h*64+cell)
        tl.store(S+h*64*128+cell*128+d,bucket+v);tl.store(Z+h*64+cell,count+1)
    else:
        k=tl.load(K+h*64+r);z+=k;s+=k[:,None]*v[None,:]
        den=tl.sum(q*z,0);out=tl.sum(q[:,None]*s,0)/den
        tl.store(S+h*64*128+r[:,None]*128+d[None,:],s);tl.store(Z+h*64+r,z)
    tl.store(O+h*128+d,out);tl.store(DEN+h,den)

class FeatureMap:
    def __init__(self, method, indices=(0,1,2,3), seed=1009, dtype=torch.float32,
                 optimized=True):
        self.method=method;self.dtype=dtype;self.m=640 if method=='favor_plus_640' else 64;self.indices=list(indices)
        self.optimized=optimized
        if method=='partition':base=PartitionFeatures(64,dtype)
        elif method=='galerkin':base=GalerkinFeatures(64,dtype)
        else:
            bank=torch.load(COMP/'results/rf_models.pt',weights_only=True)
            row=pooled_model(bank,'favor_plus') if method=='favor_plus_640' else next(r for r in bank if r['method']=='favor_plus' and r['seed']==seed)
            base=RandomFeatures(row,self.m,dtype)
        for name,val in list(vars(base).items()):
            if torch.is_tensor(val) and val.ndim and val.shape[0]==4:
                setattr(base,name,val[self.indices].contiguous())
        self.base=base;self.h=len(indices)
        if method=='favor_plus':
            self.wq_scaled=(base.wq*128**(-.25)).contiguous()
            self.wk_scaled=(base.wk*128**(-.25)).contiguous()
        if method=='partition':
            trees=json.loads((OP/'results/partitions.json').read_text())
            self.tree={}
            for side in ['q','k']:
                fields={k:[] for k in ['axis','threshold','left','right','leaf']};depth=0
                for h in self.indices:
                    ns=trees[side][h]['nodes'][:127]
                    leaves=[i for i,n in enumerate(ns) if n['left']<0 or n['left']>=127]
                    slots={node:i for i,node in enumerate(leaves)}
                    for key in ['axis','threshold','left','right']:
                        fields[key].append([(-1 if key in ['left','right'] and i in slots else n[key]) for i,n in enumerate(ns)])
                    fields['leaf'].append([slots.get(i,-1) for i in range(127)])
                    for node in leaves:
                        dd=0
                        while node:dd+=1;node=ns[node]['parent']
                        depth=max(depth,dd)
                self.tree[side]={k:torch.tensor(v,device='cuda',dtype=dtype if k=='threshold' else torch.int32) for k,v in fields.items()}
                self.tree[side]['depth']=depth

    def coordinates(self,x,side):
        b=self.base;x=x.to(self.dtype).contiguous()
        if not self.optimized or self.dtype!=torch.float32:
            return b.landmark_values(x,side)@getattr(b,'projection_'+side)
        anchor=b.k_anchor if side=='q' else b.q_anchor
        if x.shape[1]==1:
            values=torch.empty((x.shape[0],1,1024),device=x.device,dtype=x.dtype)
            _landmark_gemv[(x.shape[0],16)](x,anchor,b.scale,values,1024,64)
        else:
            logits=x@anchor.transpose(-1,-2);values=torch.empty_like(logits)
            _exp_scale[(x.shape[0],triton.cdiv(x.shape[1]*1024,256))](logits,b.scale,values,x.shape[1],1024,256)
        return values@getattr(b,'projection_'+side)

    def cells(self,x,side):
        if not self.optimized or self.dtype!=torch.float32:return self.base.cells(x,side)
        coords=self.coordinates(x,side).contiguous();t=self.tree[side]
        out=torch.empty(coords.shape[:2],device='cuda',dtype=torch.int64)
        _route[(coords.shape[0],triton.cdiv(coords.shape[1],128))](coords,t['axis'],t['threshold'],t['left'],t['right'],t['leaf'],out,
            coords.shape[1],64,127,t['depth'],128)
        return out

    def raw_features(self,x,side):
        """Original feature scale, for operator timing/equality checks only."""
        if self.method=='partition':
            cells=self.cells(x,side)
            f=F.one_hot(cells,64).to(self.dtype) if side=='q' else self.base.coefficients.transpose(-1,-2).gather(1,cells[:,:,None].expand(-1,-1,64))
            return f*self.base.sqrt_scale[:,None,None]
        if self.method=='galerkin':return self.coordinates(x,side)*self.base.sqrt_scale[:,None,None]
        if not self.optimized or self.dtype!=torch.float32 or self.m!=64:return self.base.features(x,side)
        x=x.to(self.dtype).contiguous();w=getattr(self,'w'+side+'_scaled')
        if x.shape[1]==1:
            out=torch.empty((x.shape[0],1,64),device=x.device,dtype=x.dtype)
            _favor_gemv[(x.shape[0],4)](x,w,getattr(self.base,'b'+side),out,16)
            return out
        proj=x@w;out=torch.empty_like(proj)
        _favor_exp[(x.shape[0],x.shape[1])](proj,x,getattr(self.base,'b'+side),out,x.shape[1],64,128,64,128)
        return out

    def prefill(self,q,k):
        """Returns equivalent attention factors plus any recurrent key gauge."""
        if self.method=='partition':
            cq=self.cells(q,'q');ck=self.cells(k,'k')
            # q=B[row,:], k=onehot(column) permits a one-bucket state update.
            f=self.base.coefficients.gather(1,cq[:,:,None].expand(-1,-1,64))
            return f,F.one_hot(ck,64).to(self.dtype),None
        if self.method=='galerkin':return self.coordinates(q,'q'),self.coordinates(k,'k'),None
        lq=self.log_features(q,'q');lk=self.log_features(k,'k')
        gauge=lk.amax(1)
        z=lq+gauge[:,None];z=z-z.amax(-1,keepdim=True)
        return z.exp(),(lk-gauge[:,None]).exp(),gauge

    def decode(self,q,k,state):
        if self.method=='partition':
            cq=self.cells(q,'q');ck=self.cells(k,'k')
            f=self.base.coefficients.gather(1,cq[:,:,None].expand(-1,-1,64))
            return f,ck
        if self.method=='galerkin':return self.coordinates(q,'q'),self.coordinates(k,'k')
        lq=self.log_features(q,'q');lk=self.log_features(k,'k')
        old=state['gauge'];gauge=torch.maximum(old,lk[:,0]);ratio=(old-gauge).exp()
        state['s'].mul_(ratio[:,:,None]);state['z'].mul_(ratio)
        state['gauge']=gauge
        z=lq+gauge[:,None];z=z-z.amax(-1,keepdim=True)
        return z.exp(),(lk-gauge[:,None]).exp()

    def log_features(self,x,side):
        if not self.optimized or self.dtype!=torch.float32 or self.m!=64:return self.base.log_features(x,side)
        x=x.to(self.dtype).contiguous();w=getattr(self,'w'+side+'_scaled')
        out=torch.empty((x.shape[0],x.shape[1],64),device=x.device,dtype=x.dtype)
        if x.shape[1]==1:
            _favor_gemv[(x.shape[0],4)](x,w,getattr(self.base,'b'+side),out,16,True)
        else:
            proj=x@w
            _favor_exp[(x.shape[0],x.shape[1])](proj,x,getattr(self.base,'b'+side),out,x.shape[1],64,128,64,128,True)
        return out

def causal_prefill(q,k,v,chunk=64):
    """Parallel chunk states + exact within-chunk causal products, O(N)."""
    h,n,m=q.shape;dv=v.shape[-1];pad=(-n)%chunk
    q=F.pad(q,(0,0,0,pad));k=F.pad(k,(0,0,0,pad));v=F.pad(v,(0,0,0,pad))
    qc=q.view(h,-1,chunk,m);kc=k.view(h,-1,chunk,m);vc=v.view(h,-1,chunk,dv)
    ds=kc.transpose(-1,-2)@vc;dz=kc.sum(-2)
    # Exclusive prefix without subtracting nearly equal signed totals.
    ss=F.pad(ds.cumsum(1)[:,:-1],(0,0,0,0,1,0));zz=F.pad(dz.cumsum(1)[:,:-1],(0,0,1,0))
    local=(qc@kc.transpose(-1,-2)).tril()
    num=qc@ss+local@vc;den=(qc*zz[:,:,None]).sum(-1)+local.sum(-1)
    out=(num/den[:,:,:,None]).reshape(h,-1,dv)[:,:n]
    state=dict(s=ds.sum(1),z=dz.sum(1))
    return out,den.reshape(h,-1)[:,:n],state

def recurrent_step(q,k,v,state,partition=False,optimized=True):
    if optimized and q.dtype==torch.float32 and q.shape[-1]==64:
        q=q.contiguous();k=k.contiguous();v=v.contiguous()
        state['s']=state['s'].contiguous();state['z']=state['z'].contiguous()
        out=torch.empty_like(v);den=torch.empty(q.shape[0],device=q.device,dtype=q.dtype)
        _step[(q.shape[0],)](q,k,v,state['s'],state['z'],out,den,partition)
        return out,den
    if partition:
        ix=k[:,:,None].expand(-1,-1,v.shape[-1]);state['s'].scatter_add_(1,ix,v)
        state['z'].scatter_add_(1,k,torch.ones_like(k,dtype=v.dtype))
    else:
        state['s'].add_(k[:,0,:,None]*v[:,0,None,:]);state['z'].add_(k[:,0])
    den=(q[:,0]*state['z']).sum(-1)
    out=(q@state['s'])/den[:,None,None]
    return out,den

class AttentionReplacement:
    """Split off selected Q heads before SDPA; retain shared GQA KV cache."""
    def __init__(self,original,method='teacher',seed=1009,dtype=torch.float32,optimized=True):
        self.original=original;self.method=method;self.states={};self.checks=[]
        self.maps={} if method=='teacher' else ({14:None,27:None} if method=='exact_split' else {14:FeatureMap(method,[0,1],seed,dtype,optimized),27:FeatureMap(method,[2,3],seed,dtype,optimized)})
        self.invalid_rows=0;self.nonpositive=0;self.rows=0
        self.collect_diagnostics=False;self.diagnostics=[]
    def reset(self):self.states={};self.diagnostics=[]
    def __call__(self,module,query,key,value,attention_mask,**kwargs):
        layer=module.layer_idx
        if layer not in self.maps:return self.original(module,query,key,value,attention_mask,**kwargs)
        assert query.shape[0]==1,'Current validation is batch=1, no padding.'
        n=query.shape[2];total=key.shape[2];fmap=self.maps[layer]
        heads=[0,6];remaining=[1,2,3,4,5,7,8,9,10,11];groups=module.num_key_value_groups
        # Avoid computing exact attention for the substituted heads.
        qr=query[:,remaining];kr=key[:,[h//groups for h in remaining]];vr=value[:,[h//groups for h in remaining]]
        causal=n>1 and attention_mask is None
        exact=F.scaled_dot_product_attention(qr,kr,vr,attn_mask=attention_mask,dropout_p=0.,is_causal=causal,scale=kwargs.get('scaling'))
        if self.method=='exact_split':
            y=F.scaled_dot_product_attention(query[:,heads],key,value,attn_mask=attention_mask,dropout_p=0.,is_causal=causal,scale=kwargs.get('scaling'))
            out=torch.empty_like(query);out[:,remaining]=exact;out[:,heads]=y
            return out.transpose(1,2).contiguous(),None
        q=query[0,heads].to(fmap.dtype);k=key[0,[0,1],-n:].to(fmap.dtype);v=value[0,[0,1],-n:].to(fmap.dtype)
        if n==total:
            fq,fk,gauge=fmap.prefill(q,k);y,den,state=causal_prefill(fq,fk,v)
            state['gauge']=gauge;self.states[layer]=state
        else:
            assert n==1 and layer in self.states
            fq,fk=fmap.decode(q,k,self.states[layer]);y,den=recurrent_step(fq,fk,v,self.states[layer],fmap.method=='partition')
        # Avoid CPU synchronization in timing; tensors are read only by quality eval.
        self.last_den=den;self.last_y=y
        if self.collect_diagnostics:
            self.diagnostics.append(torch.stack([(den<=0).sum(),(~torch.isfinite(y).all(-1)).sum(),torch.tensor(den.numel(),device=den.device)]))
        out=torch.empty_like(query);out[:,remaining]=exact;out[0,heads]=y.to(query.dtype)
        return out.transpose(1,2).contiguous(),None
