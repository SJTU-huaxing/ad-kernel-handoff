import sys,json,hashlib,random,argparse,time
from pathlib import Path
O=Path(__file__).resolve().parent;ROOT=O.parent;C=ROOT/'causal_direction';sys.path.insert(0,str(C))
import torch
from common import save


def select():
 if (O/'data/manifest.json').exists():return
 import pyarrow.parquet as pq
 from transformers import AutoTokenizer
 used_text=set();used_token=set()
 def scan(x):
  if isinstance(x,dict):
   if 'text_sha256'in x:used_text.add(x['text_sha256'])
   if 'token_sha256'in x:used_token.add(x['token_sha256'])
   for v in x.values():
    if isinstance(v,(dict,list)):scan(v)
  elif isinstance(x,list):
   for v in x:
    if isinstance(v,(dict,list)):scan(v)
 for f in ROOT.rglob('*manifest*.json'):
  try:scan(json.loads(f.read_text()))
  except (json.JSONDecodeError,UnicodeDecodeError):pass
 old=json.loads((C/'data/manifest.json').read_text());cache=Path('/root/autodl-tmp/hf-cache');tok=AutoTokenizer.from_pretrained(old['model'],cache_dir=str(cache),local_files_only=True);parquet=next(p for p in (cache/'datasets--EleutherAI--wikitext_document_level').rglob('*train.parquet') if 'wikitext-103-raw-v1'in str(p));pages=pq.read_table(parquet,columns=['page']).column('page').to_pylist();order=list(range(len(pages)));random.Random(911223).shuffle(order);records=[]
 for split,length,count in [('orbit_wiki',1024,64),('orbit_long',8192,16)]:
  rr=[]
  for i in order:
   text=pages[i];th=hashlib.sha256(text.encode()).hexdigest()
   if th in used_text:continue
   ids=tok(text,truncation=True,max_length=length,add_special_tokens=False)['input_ids']
   if len(ids)!=length:continue
   tt=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
   if tt in used_token:continue
   used_text.add(th);used_token.add(tt);qi=torch.randperm(length,generator=torch.Generator().manual_seed(92000+len(rr)))[:64].sort().values.tolist();rr.append(dict(split=split,index=len(rr),source_index=i,input_ids=ids,query_positions=qi,text_sha256=th,token_sha256=tt))
   if len(rr)==count:break
  assert len(rr)==count;records+=rr
 save(O/'data/manifest.json',dict(model=old['model'],revision=old['revision'],records=records,selection_seed=911223,scope='Fresh heldout relative to all text/token hashes in preceding research manifests; selected before orbit candidate evaluation.'));print(json.dumps(dict(event='selected',documents=len(records))),flush=True)

@torch.inference_mode()
def extract():
 from transformers import AutoModelForCausalLM
 from transformers.models.qwen2 import modeling_qwen2
 assert (O/'results/frozen_plan.json').exists();manifest=json.loads((O/'data/manifest.json').read_text());original=modeling_qwen2.ALL_ATTENTION_FUNCTIONS['sdpa'];capture={}
 def hook(mod,q,k,v,mask,**kw):
  if mod.layer_idx in [14,27]:capture[mod.layer_idx]=(q,k,v)
  return original(mod,q,k,v,mask,**kw)
 modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',hook);model=AutoModelForCausalLM.from_pretrained(manifest['model'],revision=manifest['revision'],cache_dir='/root/autodl-tmp/hf-cache',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.model(input_ids=torch.tensor([manifest['records'][0]['input_ids']],device='cuda'),use_cache=False)
 for split in ['orbit_wiki','orbit_long']:
  rr=[r for r in manifest['records'] if r['split']==split]
  for begin in range(0,len(rr),16):
   path=O/'data'/f'{split}_{begin//16:03d}.pt'
   if path.exists():continue
   out={k:[] for k in ['q','k','v','input_ids','query_positions']}
   for r in rr[begin:begin+16]:
    ids=torch.tensor(r['input_ids'],device='cuda');qi=torch.tensor(r['query_positions'],device='cuda');model.model(input_ids=ids[None],use_cache=False)
    for j,k in enumerate(['q','k','v']):out[k].append(torch.cat([capture[l][j][0,:,qi].cpu() if k=='q' else capture[l][j][0].cpu() for l in [14,27]]))
    out['input_ids'].append(ids.cpu());out['query_positions'].append(qi.cpu())
   torch.save({k:torch.stack(v) for k,v in out.items()},path)
  print(json.dumps(dict(event='extracted',split=split,documents=len(rr))),flush=True)
 modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['select','extract']);a=ap.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 select() if a.action=='select' else extract()
