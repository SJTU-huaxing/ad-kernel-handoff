"""Frozen confirmation evaluation; no checkpoint/hyperparameter changes."""
import argparse,gc
from common import *
from runtime import Replacement
from hybrid import Hybrid
from swa import Sliding
from transformers import AutoModelForCausalLM
from transformers.models.qwen2 import modeling_qwen2
SPLITS=['confirm_wiki','confirm_long','confirm_prose']
KINDS=['split_1','split_01','split_kl','exp_kl','softplus_kl','hedgehog','learned_prf','calibrated','gate','global_bal','global_raw','favor']
HYBRID=['split_01','exp_kl','softplus_kl','calibrated','gate','global_bal','global_raw','favor','hedgehog','learned_prf']

@torch.inference_mode()
def mixed_metrics(net,ds,scale):
 rows=[];index=KI.cuda();scale=scale.cuda().double()
 for i in range(len(ds['q'])):
  q=ds['q'][i].cuda().double();k=ds['k'][i].cuda().double()[index];v=ds['v'][i].cuda().double()[index];pos=ds['query_positions'][i].cuda();ix=torch.arange(k.shape[1],device='cuda');causal=(ix[None]<=pos[:,None])[None];remote=(ix[None]<=pos[:,None]-64)[None];local=causal&~remote
  lt=q@k.transpose(-1,-2)/math.sqrt(D)-scale[:,None,None];lp=net.log_matrix(q,k);t=lt.masked_fill(~causal,-torch.inf);p=torch.where(local,lt,lp).masked_fill(~causal,-torch.inf);a=t.softmax(-1);b=p.softmax(-1);kl=(a*(t-t.logsumexp(-1,keepdim=True)-p+p.logsumexp(-1,keepdim=True)).masked_fill(~causal,0)).sum(-1)
  y=a@v;yh=b@v;alpha=(a*local).sum(-1);beta=(b*local).sum(-1);valid=pos>=64
  if valid.any():
   at=alpha[:,valid].clamp(1e-15,1-1e-15);bt=beta[:,valid].clamp(1e-15,1-1e-15);masskl=(at*(at/bt).log()+(1-at)*((1-at)/(1-bt)).log()).mean(-1)
  else:masskl=torch.zeros(H,device='cuda',dtype=torch.float64)
  rows.append(dict(ordinal=i,kl=kl.mean(-1).tolist(),mixture_mass_kl=masskl.tolist(),output_nmse=((y-yh).square().sum((-1,-2))/y.square().sum((-1,-2)).clamp_min(1e-30)).tolist(),exact_local_mass=alpha.mean(-1).tolist()))
 return dict(summary={k:torch.tensor([r[k] for r in rows],dtype=torch.float64).mean(0).tolist() for k in rows[0] if k!='ordinal'},documents=rows,window=64,note='64-token prose is entirely in exact window; zero kernel error there does not demonstrate learned remote generalization.')

@torch.inference_mode()
def kernels():
 for split in SPLITS:
  ds=data(split)
  for kind in KINDS:
   for seed in [11,29,47]:
    name=f'{kind}_s{seed}';path=P/'results'/f'kernel_{split}_{name}.json'
    if path.exists():continue
    net,meta=load(name);scale=torch.tensor(meta['log_scale']);out=evaluate(net,ds,scale)
    if kind in HYBRID:out['hybrid']=mixed_metrics(net,ds,scale)
    save(path,dict(name=name,split=split,**out));print(json.dumps(dict(event='kernel',split=split,name=name,kl=sum(out['summary']['kl'])/H,output=sum(out['summary']['output_nmse'])/H,hybrid_output=sum(out['hybrid']['summary']['output_nmse'])/H if 'hybrid'in out else None)),flush=True)
    del net

@torch.inference_mode()
def ppl():
 assert json.loads((P/'checks/model_cache.json').read_text())['passed']
 manifest=json.loads((P/'data/manifest.json').read_text());original=modeling_qwen2.ALL_ATTENTION_FUNCTIONS['sdpa'];model=AutoModelForCausalLM.from_pretrained(manifest['model'],revision=manifest['revision'],cache_dir='/root/autodl-tmp/hf-cache',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
 cases=[('pure','teacher'),('window','swa448_sink4')]
 cases += [('pure',f'{k}_s{s}') for k in ['split_01','split_kl','exp_kl','softplus_kl','favor','hedgehog','learned_prf'] for s in [11,29,47]]
 cases += [('hybrid',f'{k}_s{s}') for k in ['split_01','softplus_kl','global_bal','global_raw','calibrated','gate','favor'] for s in [11,29,47]]
 # All candidate settings fixed above; long-context score covers all next-token
 # targets, not just selected Q rows. Same input docs, real full-model feedback.
 for mode,name in cases:
  rt=Sliding(original) if mode=='window' else (Hybrid if mode=='hybrid' else Replacement)(original,name);rt.collect=False;modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',rt)
  for split in SPLITS:
   path=P/'results'/f'ppl_{split}_{mode}_{name}.json'
   if path.exists():continue
   records=[r for r in manifest['records'] if r['split']==split];rows=[];start=time.perf_counter()
   # Consistent model warmup, never retained as a measurement.
   rt.reset();model.model(input_ids=torch.tensor([records[0]['input_ids']],device='cuda'),use_cache=False)
   for i,r in enumerate(records):
    ids=torch.tensor([r['input_ids']],device='cuda');rt.reset();hidden=model.model(input_ids=ids,use_cache=False).last_hidden_state;losses=[]
    for j in range(0,len(r['input_ids'])-1,256):
     end=min(j+256,len(r['input_ids'])-1);logits=model.lm_head(hidden[:,j:end]).float();nll=F.cross_entropy(logits[0],ids[0,j+1:end+1],reduction='none');losses.append(nll.double().sum())
    total=float(torch.stack(losses).sum());assert math.isfinite(total);rows.append(dict(ordinal=i,nll=total,tokens=len(r['input_ids'])-1))
   nll=sum(r['nll'] for r in rows);tokens=sum(r['tokens'] for r in rows);out=dict(name=name,mode=mode,split=split,ppl=math.exp(nll/tokens),nll_per_token=nll/tokens,documents=rows,seconds=time.perf_counter()-start,scope='Frozen full model; only all heads at L14,L27 changed (24/336); no model finetuning. Unpadded per-document PPL, all next-token targets. Prose64 fully exact for window/hybrid; not learned remote evidence.');save(path,out);print(json.dumps({k:v for k,v in out.items() if k!='documents'}),flush=True)
  del rt;gc.collect();torch.cuda.empty_cache()
 modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['kernels','ppl']);a=ap.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 kernels() if a.action=='kernels' else ppl()
