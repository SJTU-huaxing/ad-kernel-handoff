import sys,argparse,gc
from pathlib import Path
O=Path(__file__).resolve().parent;C=O.parent/'causal_direction';sys.path.insert(0,str(C))
from common import *
from evaluate import mixed_metrics
from runtime import Replacement,cache_for
from hybrid import Hybrid,hybrid_storage
from swa import Sliding
from transformers import AutoModelForCausalLM
from transformers.models.qwen2 import modeling_qwen2
KINDS=['split_01','split_kl','softplus_kl','favor','orbit_split_01','orbit_split_kl']

def dataset(split):
 boxes=[torch.load(f,weights_only=True) for f in sorted((O/'data').glob(split+'_*.pt'))];return {k:torch.cat([b[k] for b in boxes]) for k in boxes[0]}

@torch.inference_mode()
def kernels():
 for split in ['orbit_wiki','orbit_long']:
  ds=dataset(split)
  for k in KINDS:
   for seed in [11,29,47]:
    name=f'{k}_s{seed}';path=O/'results'/f'kernel_{split}_{name}.json'
    if path.exists():continue
    net,meta=load(name);scale=torch.tensor(meta['log_scale']);metrics=evaluate(net,ds,scale);metrics['hybrid']=mixed_metrics(net,ds,scale);save(path,dict(name=name,split=split,**metrics));print(json.dumps(dict(event='kernel',split=split,name=name,kl=sum(metrics['summary']['kl'])/H,output=sum(metrics['summary']['output_nmse'])/H,hybrid_output=sum(metrics['hybrid']['summary']['output_nmse'])/H)),flush=True);del net

@torch.inference_mode()
def ppl():
 manifest=json.loads((O/'data/manifest.json').read_text());original=modeling_qwen2.ALL_ATTENTION_FUNCTIONS['sdpa'];model=AutoModelForCausalLM.from_pretrained(manifest['model'],revision=manifest['revision'],cache_dir='/root/autodl-tmp/hf-cache',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();cases=[('pure','teacher'),('window','swa448_sink4')]+[('pure',f'{k}_s{s}') for k in KINDS for s in [11,29,47]]+[('hybrid',f'{k}_s{s}') for k in ['split_01','softplus_kl','orbit_split_01','orbit_split_kl'] for s in [11,29,47]]
 for mode,name in cases:
  rt=Sliding(original) if mode=='window' else (Hybrid if mode=='hybrid' else Replacement)(original,name);rt.collect=False;modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',rt)
  for split in ['orbit_wiki','orbit_long']:
   path=O/'results'/f'ppl_{split}_{mode}_{name}.json'
   if path.exists():continue
   records=[r for r in manifest['records'] if r['split']==split];rt.reset();model.model(input_ids=torch.tensor([records[0]['input_ids']],device='cuda'),use_cache=False);rows=[];start=time.perf_counter()
   for i,r in enumerate(records):
    ids=torch.tensor([r['input_ids']],device='cuda');rt.reset();hidden=model.model(input_ids=ids,use_cache=False).last_hidden_state;losses=[]
    for j in range(0,ids.shape[1]-1,256):
     end=min(j+256,ids.shape[1]-1);logits=model.lm_head(hidden[:,j:end]).float();losses.append(F.cross_entropy(logits[0],ids[0,j+1:end+1],reduction='sum').double())
    nll=float(torch.stack(losses).sum());assert math.isfinite(nll);rows.append(dict(ordinal=i,nll=nll,tokens=ids.shape[1]-1))
   nll=sum(r['nll'] for r in rows);tokens=sum(r['tokens'] for r in rows);out=dict(name=name,mode=mode,split=split,ppl=math.exp(nll/tokens),nll_per_token=nll/tokens,documents=rows,seconds=time.perf_counter()-start,scope='New confirmation texts, frozen full model,24/336heads at the same two complete layers. No LLM training or confirmation-based hyperparameter changes.');save(path,out);print(json.dumps({k:v for k,v in out.items() if k!='documents'}),flush=True)
  del rt;gc.collect();torch.cuda.empty_cache()
 modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['kernels','ppl']);a=ap.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 kernels() if a.action=='kernels' else ppl()
