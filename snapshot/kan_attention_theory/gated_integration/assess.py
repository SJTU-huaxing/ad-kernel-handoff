import argparse, gc, statistics
import native as n
from native import *

def install(model,family,name,norms):
 n.MAPS=None
 if name=='native':return None
 box=torch.load(P/'fits'/f'{name}.pt',weights_only=True)
 maps=adapters(model,family,box['metadata']['kind'],norms,box['metadata']['seed'])
 maps.load_state_dict(box['state_dict']);return maps

def state_bytes(obj,seen=None):
 if seen is None:seen=set()
 if id(obj) in seen:return 0
 seen.add(id(obj))
 if isinstance(obj,torch.Tensor):return obj.numel()*obj.element_size()
 if isinstance(obj,dict):return sum(state_bytes(v,seen) for v in obj.values())
 if isinstance(obj,(list,tuple)):return sum(state_bytes(v,seen) for v in obj)
 if hasattr(obj,'__dict__'):return state_bytes(vars(obj),seen)
 return 0

@torch.no_grad()
def speed(model,record):
 x=torch.tensor([record['input_ids']],device='cuda');pref=x[:,:512]
 # Warm both prefill and recurrent paths before timing.
 cache=model(pref,use_cache=True).past_key_values
 for j in range(4):cache=model(x[:,512+j:513+j],past_key_values=cache,use_cache=True).past_key_values
 torch.cuda.synchronize();p=[];d=[];cache_bytes=[]
 for _ in range(3):
  torch.cuda.synchronize();start=time.perf_counter();cache=model(pref,use_cache=True).past_key_values
  torch.cuda.synchronize();p.append((time.perf_counter()-start)*1000)
  start=time.perf_counter()
  for j in range(32):
   out=model(x[:,512+j:513+j],past_key_values=cache,use_cache=True);cache=out.past_key_values
  torch.cuda.synchronize();d.append((time.perf_counter()-start)*1000/32)
  assert torch.isfinite(out.logits).all()
  cache_bytes.append(state_bytes(cache))
 return dict(prefill_512_ms=statistics.median(p),decode_ms_per_token=statistics.median(d),
  prefill_samples_ms=p,decode_samples_ms=d,state_bytes=cache_bytes,
  scope='Batch1 BF16 native full model, 512-token prefill then 32 fixed real continuation tokens; 3 hot runs; Python/FLA reference feature adapters. Not an optimized-fusion benchmark.')

def main(family,timing=False,followup=False):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 model=model_load(family);norms=torch.load(P/'data'/f'{family}_norm.pt',weights_only=True)
 kinds=['signed_residual'] if followup else KINDS[family]
 data=data_load()['records'];names=['native']+[f'{family}_{k}_s{s}' for k in kinds for s in [11,29]]
 for name in names:
  if name!='native' and not (P/'fits'/f'{name}.pt').exists():raise RuntimeError(f'Missing frozen fit {name}')
  maps=install(model,family,name,norms)
  for split in ['validation','confirm_wiki','confirm_long']:
   path=P/'results'/f'ppl_{family}_{split}_{name}.json'
   if path.exists():continue
   docs=[r for r in data if r['split']==split]
   result=evaluate(model,docs,batch=4 if split!='confirm_long' else 1)
   save(path,dict(family=family,name=name,split=split,**result))
   print(json.dumps(dict(event='ppl',family=family,name=name,split=split,ppl=result['ppl'])),flush=True)
  if timing and (name=='native' or name.endswith('_s11')):
   path=P/'results'/f'timing_{family}_{name}.json'
   if not path.exists():
    out=speed(model,next(r for r in data if r['split']=='confirm_long'))
    save(path,dict(family=family,name=name,**out));print(json.dumps(dict(event='timing',name=name,**out)),flush=True)
  n.MAPS=None;del maps;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('family');ap.add_argument('--timing',action='store_true');ap.add_argument('--followup',action='store_true');a=ap.parse_args();main(a.family,a.timing,a.followup)
