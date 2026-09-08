"""Fixed w=64 exact recent window plus a remote rank64 state, batch1."""
from runtime import *

@torch.inference_mode()
def local_attention(q,k,v,scale,window=64,offset=0):
 h,n,d=q.shape;kg=k.repeat_interleave(6,0).float();vg=v.repeat_interleave(6,0).float();ys=[];zs=[]
 for begin in range(0,n,128):
  pos=torch.arange(begin,min(begin+128,n),device=q.device)+offset
  ix=pos[:,None]-torch.arange(window,device=q.device)[None];valid=ix>=0;ix=ix.clamp_min(0)
  logits=(q[:,begin:begin+128,None].float()*kg[:,ix]).sum(-1)/math.sqrt(d)-scale[:,None,None]
  logits=logits.masked_fill(~valid[None],-torch.inf);a=logits.softmax(-1)
  ys.append((a[:,:,:,None]*vg[:,ix]).sum(-2));zs.append(logits.logsumexp(-1))
 return torch.cat(ys,1),torch.cat(zs,1)

class Hybrid(Replacement):
 def __init__(self,*args,window=64,**kwargs):super().__init__(*args,**kwargs);self.window=window;self.windows={}
 def reset(self):super().reset();self.windows={}
 def __call__(self,module,q,k,v,mask,**kw):
  l=module.layer_idx
  if l not in self.maps:return self.original(module,q,k,v,mask,**kw)
  assert q.shape[0]==1;q=q[0];k=k[0];v=v[0];n=q.shape[1];w=self.window;f=self.maps[l]
  scale=torch.tensor(f.meta['log_scale'][0 if l==14 else 12:12 if l==14 else 24],device=q.device,dtype=torch.float32)
  if l not in self.windows:
   assert k.shape[1]==n
   yl,zl=local_attention(q,k,v,scale,w)
   if n>w:
    lq,lk=f.logs(q[:,w:],k[:,:-w]);yr,den,st=prefill(lq,lk,v[:,:-w].float().repeat_interleave(6,0));self.states[l]=st
    zr=den.log()+(lq+st['g'][:,None]).logsumexp(-1);a=(zl[:,w:]-zr).sigmoid();yl[:,w:]=a[:,:,None]*yl[:,w:]+(1-a[:,:,None])*yr
   self.windows[l]=dict(k=k[:,-w:].contiguous().clone(),v=v[:,-w:].contiguous().clone())
   y=yl
  else:
   assert n==1 and k.shape[1]==1
   win=self.windows[l];oldk=win['k'];oldv=win['v'];fullk=torch.cat([oldk,k],1);fullv=torch.cat([oldv,v],1)
   yl,zl=local_attention(q,fullk,fullv,scale,w,offset=oldk.shape[1])
   if oldk.shape[1]>=w:
    lq,lk=f.logs(q,oldk[:,:1]);vg=oldv[:,:1]
    if l in self.states:
     yr,den=step(lq,lk,vg,self.states[l],self.optimized)
     normalization=(lq[:,0]+self.states[l]['g']).amax(-1) if self.optimized else (lq[:,0]+self.states[l]['g']).logsumexp(-1)
     zr=den.log()+normalization[:,None]
    else:
     yr,den,self.states[l]=prefill(lq,lk,vg.float().repeat_interleave(6,0));zr=den.log()+(lq+self.states[l]['g'][:,None]).logsumexp(-1)
    a=(zl-zr).sigmoid();y=a[:,:,None]*yl+(1-a[:,:,None])*yr
   else:y=yl
   self.windows[l]=dict(k=fullk[:,-w:].contiguous().clone(),v=fullv[:,-w:].contiguous().clone())
  if self.collect:self.diagnostics.append(torch.stack([(~torch.isfinite(y)).sum()]))
  return y[None].transpose(1,2).to(q.dtype).contiguous(),None

def hybrid_storage(cache,runtime):
 out=storage(cache,runtime);window=sum(t.untyped_storage().nbytes() for win in getattr(runtime,'windows',{}).values() for t in win.values());out['window_bytes']=window;out['total_bytes']+=window;return out
