"""Capture real post-RoPE Q/K for complete GQA layers and full causal rows."""
import json,hashlib,random,sys,time,math,argparse
from pathlib import Path
import torch
P=Path(__file__).resolve().parent;ROOT=P.parent;OLD=ROOT/'single_pass_mulkan/data';CACHE=Path('/root/autodl-tmp/hf-cache')
def save(path,x):path.write_text(json.dumps(x,indent=2,allow_nan=False))
def select():
 from transformers import AutoTokenizer
 from datasets import Dataset
 import pyarrow.parquet as pq
 old=json.loads((OLD/'manifest.json').read_text());prior=json.loads((ROOT/'real_llm_pilot/data_qwen25_1p5b/manifest.json').read_text())
 used=old['documents']+prior['documents']
 for f in (ROOT/'mlp_direction/results').glob('fresh_manifest*.json'):used+=json.loads(f.read_text())['records']
 text_hash={r['text_sha256'] for r in used};tok_hash={r['token_sha256'] for r in used}
 tok=AutoTokenizer.from_pretrained(old['model'],cache_dir=str(CACHE),local_files_only=True)
 records=[]
 # Reuse fixed training/validation source documents; select new Q positions.
 for split in ['train','validation']:
  sr=[r for r in old['shards'] if r['split']==split]
  docs=torch.cat([torch.load(OLD/r['file'],weights_only=True)['input_ids'] for r in sr])
  mr=[r for r in old['documents'] if r['split']==split]
  assert len(docs)==len(mr)
  for i,(ids,meta) in enumerate(zip(docs,mr)):
   qi=torch.randperm(len(ids),generator=torch.Generator().manual_seed(81000+i))[:64].sort().values
   records.append(dict(split=split,index=i,input_ids=ids.tolist(),query_positions=qi.tolist(),text_sha256=meta['text_sha256'],token_sha256=meta['token_sha256']))
 parquet=next(p for p in (CACHE/'datasets--EleutherAI--wikitext_document_level').rglob('*train.parquet') if 'wikitext-103-raw-v1' in str(p))
 pages=pq.read_table(parquet,columns=['page']).column('page').to_pylist();order=list(range(len(pages)));random.Random(810917).shuffle(order)
 for split,length,count in [('confirm_wiki',1024,128),('confirm_long',8192,24)]:
  chosen=[]
  for i in order:
   txt=pages[i];th=hashlib.sha256(txt.encode()).hexdigest()
   if th in text_hash:continue
   ids=tok(txt,truncation=True,max_length=length,add_special_tokens=False)['input_ids']
   if len(ids)!=length:continue
   hh=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
   if hh in tok_hash:continue
   text_hash.add(th);tok_hash.add(hh)
   qi=torch.randperm(length,generator=torch.Generator().manual_seed(82000+len(chosen)))[:64].sort().values.tolist()
   chosen.append(dict(split=split,index=len(chosen),source_index=i,input_ids=ids,query_positions=qi,text_sha256=th,token_sha256=hh))
   if len(chosen)==count:break
  assert len(chosen)==count,(split,len(chosen));records+=chosen
 # Independent book-prose prompts, new in this line of research. 64-token
 # prefixes, no claim that different prompts are independent source books.
 source=next((CACHE/'datasets/EleutherAI___lambada_openai').rglob('*test.arrow'))
 ds=Dataset.from_file(str(source));order=list(range(len(ds)));random.Random(83019).shuffle(order);chosen=[]
 for i in order:
  txt=ds[i]['text'];th=hashlib.sha256(txt.encode()).hexdigest()
  if th in text_hash:continue
  ids=tok(txt,add_special_tokens=False)['input_ids']
  if len(ids)<64:continue
  ids=ids[:64];hh=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
  if hh in tok_hash:continue
  qi=torch.randperm(64,generator=torch.Generator().manual_seed(84000+len(chosen)))[:64].sort().values.tolist()
  chosen.append(dict(split='confirm_prose',index=len(chosen),source_index=i,input_ids=ids,query_positions=qi,text_sha256=th,token_sha256=hh));text_hash.add(th);tok_hash.add(hh)
  if len(chosen)==128:break
 assert len(chosen)==128,len(chosen);records+=chosen
 save(P/'data/manifest.json',dict(model=old['model'],revision=old['model_revision'],records=records,layers=[14,27],query_heads_per_layer=12,key_heads_per_layer=2,dim=128,training_queries_per_doc=64,
  scope='Complete layer14 and layer27 GQA groups, 24/336 Q heads; no LLM finetuning. Full causal key prefixes for uniformly selected distinct Q. All configurations to be frozen before confirmation evaluation.'))
 print(json.dumps(dict(event='selected',counts={s:sum(r['split']==s for r in records) for s in sorted({r['split'] for r in records})})),flush=True)

@torch.inference_mode()
def extract(confirm=False):
 from transformers import AutoModelForCausalLM
 from transformers.models.qwen2 import modeling_qwen2
 m=json.loads((P/'data/manifest.json').read_text());original=modeling_qwen2.ALL_ATTENTION_FUNCTIONS['sdpa'];captured={}
 def capture(module,q,k,v,mask,**kw):
  if module.layer_idx in [14,27]:captured[module.layer_idx]=(q,k,v)
  return original(module,q,k,v,mask,**kw)
 modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',capture)
 model=AutoModelForCausalLM.from_pretrained(m['model'],revision=m['revision'],cache_dir=str(CACHE),local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
 # Warm up once before retaining activations, to keep the first shard's
 # low-precision path consistent with subsequent captures.
 model.model(input_ids=torch.tensor([m['records'][0]['input_ids']],device='cuda'),use_cache=False)
 splits=['confirm_wiki','confirm_long','confirm_prose'] if confirm else ['train','validation']
 if confirm:assert (P/'results/frozen_plan.json').exists()
 for split in splits:
  rows=[r for r in m['records'] if r['split']==split];start=time.perf_counter()
  for begin in range(0,len(rows),32):
   path=P/'data'/f'{split}_{begin//32:03d}.pt'
   if path.exists():continue
   part={s:[] for s in ['q','k','input_ids','query_positions']}
   if split!='train':part['v']=[]
   for r in rows[begin:begin+32]:
    ids=torch.tensor(r['input_ids'],device='cuda');qi=torch.tensor(r['query_positions'],device='cuda')
    model.model(input_ids=ids[None],use_cache=False)
    part['q'].append(torch.cat([captured[l][0][0,:,qi].cpu() for l in [14,27]]))
    part['k'].append(torch.cat([captured[l][1][0].cpu() for l in [14,27]]))
    if 'v' in part:part['v'].append(torch.cat([captured[l][2][0].cpu() for l in [14,27]]))
    part['input_ids'].append(ids.cpu());part['query_positions'].append(qi.cpu())
   torch.save({k:torch.stack(v) for k,v in part.items()},path)
   if begin%512==0:print(json.dumps(dict(event='extract',split=split,documents=min(begin+32,len(rows)),seconds=time.perf_counter()-start)),flush=True)
  print(json.dumps(dict(event='extracted',split=split,documents=len(rows),seconds=time.perf_counter()-start)),flush=True)
 modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['select','extract']);ap.add_argument('--confirm',action='store_true');a=ap.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 select() if a.action=='select' else extract(a.confirm)
