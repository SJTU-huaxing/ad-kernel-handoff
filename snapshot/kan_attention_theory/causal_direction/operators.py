"""Stable chunk scan and fused log-feature decode; no KV-history allocation."""
from common import *
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def _baseline_log(Q,K,W,B,OUT,H:tl.constexpr,KIND:tl.constexpr):
 h=tl.program_id(0);side=tl.program_id(1);d=tl.arange(0,128);r=tl.arange(0,64)
 if side==0:x=tl.load(Q+h*128+d).to(tl.float32)
 else:x=tl.load(K+(h//6)*128+d).to(tl.float32)
 x=x*0.29730177875068026
 if KIND==0:
  w=tl.load(W+(h*64+r[:,None])*128+d[None,:]);p=tl.sum(w*x[None,:],1)-.5*tl.sum(x*x,0)+.5*tl.load(B+h*64+r)-2.0794415416798357
 else:
  rr=r%32;w=tl.load(W+(h*32+rr[:,None])*128+d[None,:]);z=tl.sum(w*x[None,:],1)+tl.load(B+h*32+rr);p=tl.where(r<32,z,-z)-2.0794415416798357
 tl.store(OUT+(side*H+h)*64+r,p)

@triton.jit
def _mlp_log(Q,K,W1,W2,B,MEAN,STD,OUT,H:tl.constexpr,VAR:tl.constexpr):
 h=tl.program_id(0);side=tl.program_id(1);s=side*H+h
 d=tl.arange(0,128);a=tl.arange(0,256);r=tl.arange(0,64)
 if side==0:x=tl.load(Q+h*128+d)
 else:x=tl.load(K+(h//6)*128+d)
 x=(x-tl.load(MEAN+s*128+d))/tl.load(STD+s*128+d)
 w=tl.load(W1+(s*192+a[:,None])*128+d[None,:],a[:,None]<192,other=0)
 z=tl.sum(w*x[None,:],1);hidden=z*tl.sigmoid(z)
 w2=tl.load(W2+(s*64+r[:,None])*192+a[None,:],a[None,:]<192,other=0)
 zz=tl.sum(w2*hidden[None,:],1)+tl.load(B+s*64+r)
 if VAR==0:zz=zz-2.0794415416798357
 elif VAR==1 or (VAR==3 and side==0):
  amp=tl.sum(tl.where(r==63,zz,0),0);zz=tl.where(r<63,zz,0)
  mx=tl.max(zz,0);zz=zz-mx-tl.log(tl.sum(tl.exp(zz-mx),0))+amp+2.0794415416798357
 elif VAR==3:zz=zz-2.0794415416798357
 else:
  soft=tl.maximum(zz,0)+libdevice.log1p(tl.exp(-tl.abs(zz)))
  zz=tl.where(zz< -20,zz,tl.log(soft))+0.36651292058166435-2.0794415416798357
 tl.store(OUT+(side*H+h)*64+r,zz)

@triton.jit
def _step_log(LQ,LK,V,S,Z,G,O,DEN):
 h=tl.program_id(0);r=tl.arange(0,64);d=tl.arange(0,128)
 lq=tl.load(LQ+h*64+r);lk=tl.load(LK+h*64+r);old=tl.load(G+h*64+r)
 gauge=tl.maximum(old,lk);rescale=tl.exp(old-gauge)
 qq=lq+gauge;qq=tl.exp(qq-tl.max(qq,0));kk=tl.exp(lk-gauge)
 vv=tl.load(V+(h//6)*128+d)
 ss=tl.load(S+h*64*128+r[:,None]*128+d[None,:])*rescale[:,None]+kk[:,None]*vv[None,:]
 zz=tl.load(Z+h*64+r)*rescale+kk
 den=tl.sum(qq*zz,0);yy=tl.sum(qq[:,None]*ss,0)/den
 tl.store(S+h*64*128+r[:,None]*128+d[None,:],ss);tl.store(Z+h*64+r,zz);tl.store(G+h*64+r,gauge)
 tl.store(O+h*128+d,yy);tl.store(DEN+h,den)

@torch.inference_mode()
def prefill(lq,lk,v,state=None,chunk=64):
 h,n,m=lq.shape;d=v.shape[-1];g=lk.amax(1)
 if state is not None:g=torch.maximum(g,state['g'])
 k=(lk-g[:,None]).exp();q=(lq+g[:,None]).softmax(-1)
 count=(n+chunk-1)//chunk;pad=count*chunk-n
 q=F.pad(q,(0,0,0,pad)).reshape(h,count,chunk,m)
 k=F.pad(k,(0,0,0,pad)).reshape(h,count,chunk,m)
 v=F.pad(v,(0,0,0,pad)).reshape(h,count,chunk,d)
 sums=(k.transpose(-1,-2)@v).cumsum(1);zsum=k.sum(-2).cumsum(1)
 previous=F.pad(sums[:,:-1],(0,0,0,0,1,0));zprevious=F.pad(zsum[:,:-1],(0,0,1,0))
 if state is not None:
  fac=(state['g']-g).exp();ss=state['s']*fac[:,:,None];zz=state['z']*fac
  previous=previous+ss[:,None];zprevious=zprevious+zz[:,None];sums=sums+ss[:,None];zsum=zsum+zz[:,None]
 local=(q@k.transpose(-1,-2)).tril()
 numerator=q@previous+local@v;den=(q*zprevious[:,:,None]).sum(-1)+local.sum(-1)
 y=numerator/den.clamp_min(torch.finfo(den.dtype).tiny)[...,None]
 return y.reshape(h,count*chunk,d)[:,:n].contiguous(),den.reshape(h,count*chunk)[:,:n],dict(s=sums[:,-1].contiguous(),z=zsum[:,-1].contiguous(),g=g.contiguous())

@torch.inference_mode()
def step(lq,lk,vg,state,optimized=True):
 h=lq.shape[0]
 if optimized and lq.dtype==torch.float32:
  out=torch.empty(h,1,128,device=lq.device);den=torch.empty(h,1,device=lq.device)
  _step_log[(h,)](lq.contiguous(),lk.contiguous(),vg.contiguous(),state['s'],state['z'],state['g'],out,den,num_warps=4)
  return out,den
 v=vg.repeat_interleave(6,0);g=torch.maximum(state['g'],lk[:,0]);fac=(state['g']-g).exp()
 q=(lq[:,0]+g).softmax(-1);k=(lk[:,0]-g).exp()
 state['s'].mul_(fac[:,:,None]).add_(k[:,:,None]*v[:,0,None,:]);state['z'].mul_(fac).add_(k);state['g']=g
 den=(q*state['z']).sum(-1);return ((q[:,:,None]*state['s']).sum(1)/den[:,None])[:,None],den[:,None]

class LayerFeatures:
 def __init__(self,name,layer,dtype=torch.float32,optimized=True):
  net,meta=load(name,dtype);start=0 if layer==14 else 12
  for mod in net.modules():
   for k,v in list(mod._parameters.items()):
    if v is not None:mod._parameters[k]=nn.Parameter(v[start:start+12].clone(memory_format=torch.contiguous_format),requires_grad=False)
   for k,v in list(mod._buffers.items()):
    if v is not None and v.ndim and v.shape[0]==24:mod._buffers[k]=v[start:start+12].clone(memory_format=torch.contiguous_format)
  self.net=net;self.meta=meta;self.dtype=dtype;self.optimized=optimized
  if hasattr(net,'qnet'):
   self.w1=torch.stack([net.qnet.w1,net.knet.w1]);self.w2=torch.stack([net.qnet.w2,net.knet.w2]);self.bias=torch.stack([net.qnet.bias,net.knet.bias]);self.mean=torch.stack([net.q_mean,net.k_mean]);self.std=torch.stack([net.q_std,net.k_std])
   self.var=0 if net.variant=='exp' else 1 if net.variant=='factorized_both' else 3 if net.variant=='gauge_calibrated' else 2
   # Let reference GEMM and fused GEMV share one packed parameter storage.
   for side,j in [('q',0),('k',1)]:
    mod=getattr(net,side+'net');mod.w1=nn.Parameter(self.w1[j],requires_grad=False);mod.w2=nn.Parameter(self.w2[j],requires_grad=False);mod.bias=nn.Parameter(self.bias[j],requires_grad=False)
    setattr(net,side+'_mean',self.mean[j]);setattr(net,side+'_std',self.std[j])
 def logs(self,q,k):
  q=q.contiguous();k=k.contiguous()
  if self.optimized and self.dtype==torch.float32 and q.shape[1]==1 and hasattr(self,'w1'):
   out=torch.empty(2,12,1,64,device=q.device)
   _mlp_log[(12,2)](q,k,self.w1,self.w2,self.bias,self.mean,self.std,out,12,self.var,num_warps=8,enable_fp_fusion=False)
   return out[0],out[1]
  if self.optimized and self.dtype==torch.float32 and q.shape[1]==1 and hasattr(self.net,'weight'):
   out=torch.empty(2,12,1,64,device=q.device)
   _baseline_log[(12,2)](q,k,self.net.weight,self.net.bias,out,12,1 if self.net.kind=='hedgehog' else 0,num_warps=4,enable_fp_fusion=False)
   return out[0],out[1]
  return self.net.log_feature(q.to(self.dtype),'q'),self.net.log_feature(k.to(self.dtype).repeat_interleave(6,0),'k')
