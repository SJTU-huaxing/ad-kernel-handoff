"""Replace two complete GQA layers and genuinely omit their KV histories."""
from common import *
from operators import LayerFeatures,prefill,step
from transformers.cache_utils import DynamicCache,DynamicLayer

class StateOnlyLayer(DynamicLayer):
 def __init__(self):super().__init__();self.seen=0
 def update(self,key_states,value_states,*args,**kwargs):
  if not self.is_initialized:
   self.dtype=key_states.dtype;self.device=key_states.device
   # Fresh empty storage: a zero-length VIEW could retain the full tensor.
   self.keys=torch.empty(0,device=self.device,dtype=self.dtype);self.values=torch.empty_like(self.keys);self.is_initialized=True
  self.seen+=key_states.shape[-2]
  return key_states,value_states
 def get_seq_length(self):return self.seen
 def reset(self):self.seen=0

def cache_for(model,linear=True):
 cache=DynamicCache(config=model.config)
 if linear:
  for l in [14,27]:cache.layers[l]=StateOnlyLayer()
 return cache

class Replacement:
 def __init__(self,original,name,dtype=torch.float32,optimized=True):
  self.original=original;self.name=name;self.maps={} if name=='teacher' else {l:LayerFeatures(name,l,dtype,optimized) for l in [14,27]};self.states={};self.diagnostics=[];self.collect=True;self.optimized=optimized
 def reset(self):self.states={};self.diagnostics=[]
 def __call__(self,module,q,k,v,mask,**kw):
  l=module.layer_idx
  if l not in self.maps:return self.original(module,q,k,v,mask,**kw)
  assert q.shape[0]==1,'This pilot implements batch1, unpadded prompts.'
  f=self.maps[l];n=q.shape[2];q=q[0];k=k[0];v=v[0]
  # With StateOnlyLayer these are only the newly projected tokens; use_cache
  # False also supplies the complete prompt once. No old KV is retained.
  assert k.shape[1]==n
  lq,lk=f.logs(q,k)
  if n==1 and l in self.states:y,den=step(lq,lk,v if self.optimized and f.dtype==torch.float32 else v.to(f.dtype),self.states[l],self.optimized)
  else:y,den,self.states[l]=prefill(lq,lk,v.to(f.dtype).repeat_interleave(6,0),self.states.get(l))
  if self.collect:self.diagnostics.append(torch.stack([(den<=0).sum(),(~torch.isfinite(y)).sum()]))
  return y[None].transpose(1,2).to(q.dtype).contiguous(),None

def storage(cache,runtime):
 kv=sum(t.untyped_storage().nbytes() for l in cache.layers for t in [l.keys,l.values] if t is not None)
 st=sum(t.untyped_storage().nbytes() for s in runtime.states.values() for t in s.values())
 return dict(kv_bytes=kv,state_bytes=st,total_bytes=kv+st)
