"""Counterbalanced, same-process full-model latency; all checkpoints remain fixed."""
import statistics
import native as n
from native import *
from assess import install,state_bytes

@torch.no_grad()
def main(family):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 model=model_load(family);norms=torch.load(P/'data'/f'{family}_norm.pt',weights_only=True)
 kinds=KINDS[family]+['signed_residual']
 names=['native']+[f'{family}_{k}_s11' for k in kinds]
 maps={name:install(model,family,name,norms) for name in names}
 record=next(r for r in data_load()['records'] if r['split']=='confirm_long')
 x=torch.tensor([record['input_ids']],device='cuda');pref=x[:,:512]
 for name in names:
  n.MAPS=maps[name];cache=model(pref,use_cache=True).past_key_values
  for j in range(4):cache=model(x[:,512+j:513+j],past_key_values=cache,use_cache=True).past_key_values
 torch.cuda.synchronize();rows=[]
 for repeat in range(5):
  order=names[repeat:]+names[:repeat]
  for name in order:
   n.MAPS=maps[name]
   torch.cuda.synchronize();start=time.perf_counter()
   cache=model(pref,use_cache=True).past_key_values
   torch.cuda.synchronize();pm=(time.perf_counter()-start)*1000
   start=time.perf_counter()
   for j in range(16):cache=model(x[:,512+j:513+j],past_key_values=cache,use_cache=True).past_key_values
   torch.cuda.synchronize();dm=(time.perf_counter()-start)*1000/16
   row=dict(name=name,repeat=repeat,prefill_ms=pm,decode_ms=dm,state_bytes=state_bytes(cache))
   rows.append(row);print(json.dumps(row),flush=True)
 summary=[]
 for name in names:
  part=[r for r in rows if r['name']==name];ratios={key:[] for key in ['prefill_ms','decode_ms']}
  for r in part:
   base=next(b for b in rows if b['name']=='native' and b['repeat']==r['repeat'])
   for key in ratios:ratios[key].append(r[key]/base[key])
  summary.append(dict(name=name,
   prefill_ms=statistics.median(r['prefill_ms'] for r in part),
   decode_ms=statistics.median(r['decode_ms'] for r in part),
   prefill_ratio_to_native=statistics.median(ratios['prefill_ms']),
   decode_ratio_to_native=statistics.median(ratios['decode_ms']),
   state_bytes=part[0]['state_bytes']))
 save(P/'results'/f'balanced_timing_{family}.json',dict(family=family,rows=rows,summary=summary,
  scope='RTX3090, batch1, frozen BF16 base/FP32 adapter, TF32 disabled, no autocast; 512-token prefill and 16 fixed continuation tokens; five cyclic method orders after warmup; CUDA synchronized wall clock. Reference Python/FLA, not fused production performance.'))
 n.MAPS=None

if __name__=='__main__':main(sys.argv[1])
