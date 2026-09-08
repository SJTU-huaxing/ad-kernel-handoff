"""New query positions for a single additional calibration pass; shared keys."""
from common import *
from transformers import AutoModelForCausalLM
from transformers.models.qwen2 import modeling_qwen2

@torch.inference_mode()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 manifest=json.loads((P/'data/manifest.json').read_text());docs=[r for r in manifest['records'] if r['split']=='train']
 original=modeling_qwen2.ALL_ATTENTION_FUNCTIONS['sdpa'];capture={}
 def hook(mod,q,k,v,mask,**kw):
  if mod.layer_idx in [14,27]:capture[mod.layer_idx]=q
  return original(mod,q,k,v,mask,**kw)
 modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',hook)
 model=AutoModelForCausalLM.from_pretrained(manifest['model'],revision=manifest['revision'],cache_dir='/root/autodl-tmp/hf-cache',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
 model.model(input_ids=torch.tensor([docs[0]['input_ids']],device='cuda'),use_cache=False);start=time.perf_counter()
 for begin in range(0,len(docs),32):
  path=P/'data'/f'calibration_{begin//32:03d}.pt'
  if path.exists():continue
  qs=[];positions=[]
  for r in docs[begin:begin+32]:
   old=set(r['query_positions']);perm=torch.randperm(1024,generator=torch.Generator().manual_seed(81000+r['index'])).tolist()
   qi=torch.tensor(sorted([i for i in perm if i not in old and i>=64][:64]),device='cuda');assert len(qi)==64 and not set(qi.tolist())&old
   model.model(input_ids=torch.tensor([r['input_ids']],device='cuda'),use_cache=False)
   qs.append(torch.cat([capture[l][0,:,qi].cpu() for l in [14,27]]));positions.append(qi.cpu())
  torch.save(dict(q=torch.stack(qs),query_positions=torch.stack(positions)),path)
  if begin%512==0:print(json.dumps(dict(event='calibration_extraction',documents=begin+32,seconds=time.perf_counter()-start)),flush=True)
 save(P/'checks/calibration_data.json',dict(disjoint_from_direction_queries=True,queries_per_document=64,documents=4096,seconds=time.perf_counter()-start))
 modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

if __name__=='__main__':main()
