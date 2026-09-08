"""Training-free sliding exact attention: 448 recent positions + 4 sinks."""
from runtime import *
class Sliding:
 def __init__(self,original,window=448,sinks=4):self.original=original;self.window=window;self.sinks=sinks;self.windows={};self.states={};self.collect=False;self.diagnostics=[];self.name='swa448_sink4'
 def reset(self):self.windows={};self.states={};self.diagnostics=[]
 def __call__(self,module,q,k,v,mask,**kw):
  l=module.layer_idx
  if l not in [14,27]:return self.original(module,q,k,v,mask,**kw)
  assert q.shape[0]==1;q=q[0];k=k[0];v=v[0];n=q.shape[1];w=self.window;s=self.sinks
  if l not in self.windows:
   ys=[]
   for begin in range(0,n,128):
    end=min(begin+128,n);left=max(0,begin-w+1);ix=torch.cat([torch.arange(min(left,s),device=q.device),torch.arange(left,end,device=q.device)])
    qp=torch.arange(begin,end,device=q.device);allowed=(ix[None]<=qp[:,None])&((ix[None]<s)|(ix[None]>qp[:,None]-w))
    ys.append(F.scaled_dot_product_attention(q[:,begin:end],k[:,ix].repeat_interleave(6,0),v[:,ix].repeat_interleave(6,0),attn_mask=allowed[None],dropout_p=0.,is_causal=False))
   y=torch.cat(ys,1);ix=torch.cat([torch.arange(min(max(n-w,0),s),device=q.device),torch.arange(max(0,n-w),n,device=q.device)])
   self.windows[l]=dict(k=k[:,ix].contiguous(),v=v[:,ix].contiguous())
  else:
   assert n==1;win=self.windows[l];kk=torch.cat([win['k'],k],1);vv=torch.cat([win['v'],v],1)
   if kk.shape[1]>w+s:kk=torch.cat([kk[:,:s],kk[:,-w:]],1);vv=torch.cat([vv[:,:s],vv[:,-w:]],1)
   y=F.scaled_dot_product_attention(q,kk.repeat_interleave(6,0),vv.repeat_interleave(6,0),dropout_p=0.,is_causal=False);self.windows[l]=dict(k=kk.contiguous(),v=vv.contiguous())
  return y[None].transpose(1,2).contiguous(),None
