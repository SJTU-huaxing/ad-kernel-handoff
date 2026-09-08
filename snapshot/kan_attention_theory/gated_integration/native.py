"""Local feature-map adapters; original FLA model operations remain intact."""
import json, math, hashlib, time, sys, abc
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import fla
from transformers import AutoModelForCausalLM, AutoTokenizer
import fla.layers.gla as gla_module
import fla.layers.gated_deltanet as gdn_module
from fla.models.gated_deltanet.modeling_gated_deltanet import GatedDeltaNetForCausalLM
from fla.models.utils import FLALayer

P = Path(__file__).resolve().parent
ROOT = P.parent
CACHE = '/root/autodl-tmp/hf-cache'
LAYERS = [11, 23]
FAMILY_LAYERS = {'gla':[11,23],'gdn':[10,20]}
SPECS = {
 'gla': ('fla-hub/gla-340M-15B','6e04029dc090a2c55df712f18814db80aa39894f'),
 'gdn': ('puigde/gated-deltanet-360M-15B-slimpajama','1d1a7bf99323601635f3d63f67c97614f9c2ec10'),
}
KINDS = {'gla':['ad_shape','ad_full','exp_mlp'],
         'gdn':['ad_norm','ad_write','exp_mlp']}
CURRENT = None
MAPS = None
CAPTURE = None
ORIGINAL = {}

def save(path, obj):
 Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))

def patch_kernels():
 for family, module, names in [
  ('gla',gla_module,['chunk_gla','fused_recurrent_gla','fused_chunk_gla']),
  ('gdn',gdn_module,['chunk_gated_delta_rule','fused_recurrent_gated_delta_rule'])]:
  for name in names:
   if (family,name) in ORIGINAL: continue
   fn=getattr(module,name); ORIGINAL[family,name]=fn
   def wrapped(*args,_fn=fn,_family=family,**kw):
    if CURRENT in LAYERS:
     if CAPTURE is not None:
      CAPTURE[CURRENT]={k:kw[k].detach().cpu() for k in ['q','k','v']}
      for key in ['g','gk','beta','A_log','dt_bias']:
       if key in kw and kw[key] is not None: CAPTURE[CURRENT][key]=kw[key].detach().cpu()
     if MAPS is not None and str(CURRENT) in MAPS:
      kw=kw.copy()
      kw['q'],kw['k'],kw['v']=MAPS[str(CURRENT)](kw['q'],kw['k'],kw['v'])
    return _fn(*args,**kw)
   setattr(module,name,wrapped)

def model_load(family):
 global MAPS,CAPTURE
 MAPS=None; CAPTURE=None
 LAYERS[:]=FAMILY_LAYERS[family]
 patch_kernels()
 # Compatibility with Transformers>=5 tied-parameter mapping; no weight changes.
 GatedDeltaNetForCausalLM._tied_weights_keys={'lm_head.weight':'model.embeddings.weight'}
 # Recent Transformers renamed the abstract cache-length interface.
 FLALayer.get_max_length=FLALayer.get_max_cache_shape
 abc.update_abstractmethods(FLALayer)
 repo,rev=SPECS[family]
 model,info=AutoModelForCausalLM.from_pretrained(repo,revision=rev,cache_dir=CACHE,
  local_files_only=True,dtype=torch.bfloat16,output_loading_info=True)
 info={k:list(v) if isinstance(v,set) else v for k,v in info.items()}
 save(P/'checks'/f'{family}_load.json',dict(repository=repo,revision=rev,loading=info,
  parameter_count=sum(x.numel() for x in model.parameters()),
  compatibility_patch='Process-local: GDN tied weight list -> explicit lm_head.weight:model.embeddings.weight mapping; FLALayer.get_max_length -> existing get_max_cache_shape. No parameter changes.'))
 assert not any(info.values()),info
 model=model.cuda().eval()
 for p in model.parameters():p.requires_grad_(False)
 def pre(mod,args):
  global CURRENT
  CURRENT=mod.layer_idx
 def post(mod,args,output):
  global CURRENT
  CURRENT=None
 for layer in model.model.layers:
  layer.attn.register_forward_pre_hook(pre)
  layer.attn.register_forward_hook(post)
 return model

class FeatureMaps(nn.Module):
 def __init__(self,family,kind,norm,seed):
  super().__init__();self.family=family;self.kind=kind
  self.heads,self.d=norm['q_mean'].shape;self.m=self.d;h=192
  gen=torch.Generator().manual_seed(seed)
  for key,val in norm.items():self.register_buffer(key,val.float())
  for side in ['q','k']:
   self.register_parameter(side+'w1',nn.Parameter(torch.randn(self.heads,self.d,h,generator=gen)/math.sqrt(self.d)))
   self.register_parameter(side+'w2',nn.Parameter(torch.randn(self.heads,h,self.m,generator=gen)*.05/math.sqrt(h)))
 def coordinates(self,x,side):
  b,t,h,d=x.shape
  x=(x.float()-getattr(self,side+'_mean')[None,None])/getattr(self,side+'_std')[None,None]
  x=x.permute(2,0,1,3).reshape(h,b*t,d)
  z=F.silu(x@getattr(self,side+'w1'))@getattr(self,side+'w2')
  return z.reshape(h,b,t,self.m).permute(1,2,0,3).float()
 def feature(self,z):
  if self.kind=='exp_mlp':
   return (4*torch.tanh(z/4)).exp()/math.sqrt(self.m),None
  shape=torch.cat([z[...,:-1],torch.zeros_like(z[...,-1:])],-1)
  return shape.softmax(-1)*math.sqrt(self.m),(4*torch.tanh(z[...,-1:]/4)).exp()
 def forward(self,q,k,v):
  qd,kd,vd=q.dtype,k.dtype,v.dtype
  if self.kind=='signed_residual':
   return (q.float()+self.coordinates(q,'q')).to(qd),(k.float()+self.coordinates(k,'k')).to(kd),v
  qf,aq=self.feature(self.coordinates(q,'q'));kf,ak=self.feature(self.coordinates(k,'k'))
  if self.kind in ['ad_full','ad_norm']:
   qf=qf*aq;kf=kf*ak
  elif self.kind=='ad_write':
   # L2-normalized keys erase with beta; only the write value gets the amplitude.
   v=v.float()*ak
  return qf.to(qd),kf.to(kd),v.to(vd)

def adapters(model,family,kind,norms,seed):
 global MAPS
 MAPS=nn.ModuleDict({str(l):FeatureMaps(family,kind,norms[str(l)],seed+1000*l).cuda() for l in LAYERS})
 return MAPS

@torch.no_grad()
def collect_norm(model,family,docs):
 global CAPTURE
 path=P/'data'/f'{family}_norm.pt'
 if path.exists():return torch.load(path,weights_only=True)
 acc={str(l):{s:[] for s in ['q','k']} for l in LAYERS}
 samples=[]
 for i in range(32):
  CAPTURE={}
  model(torch.tensor([docs[i]['input_ids']],device='cuda'),use_cache=False,logits_to_keep=1)
  for l in LAYERS:
   for s in ['q','k']:acc[str(l)][s].append(CAPTURE[l][s].float())
  if i<2:samples.append(CAPTURE)
 CAPTURE=None;norms={}
 for l in LAYERS:
  norms[str(l)]={}
  for s in ['q','k']:
   x=torch.cat(acc[str(l)][s],0).double()
   norms[str(l)][s+'_mean']=x.mean((0,1)).float()
   norms[str(l)][s+'_std']=x.std((0,1)).clamp_min(.03).float()
 torch.save(norms,path);torch.save(samples,P/'data'/f'{family}_operator_samples.pt')
 return norms

def data_load():return json.loads((P/'data/manifest.json').read_text())

@torch.no_grad()
def evaluate(model,docs,batch=4):
 rows=[]
 for start in range(0,len(docs),batch):
  part=docs[start:start+batch]
  x=torch.tensor([r['input_ids'] for r in part],device='cuda')
  logits=model(x,use_cache=False).logits
  ce=F.cross_entropy(logits[:,:-1].float().flatten(0,1),x[:,1:].flatten(),reduction='none').reshape(len(part),-1)
  for r,c in zip(part,ce):rows.append(dict(index=r['index'],nll=float(c.sum()),tokens=c.numel()))
 nll=sum(r['nll'] for r in rows);tokens=sum(r['tokens'] for r in rows)
 return dict(ppl=math.exp(nll/tokens),nll_per_token=nll/tokens,documents=rows)

def manifest_prepare():
 path=P/'data/manifest.json'
 if path.exists():return
 old=json.loads((ROOT/'causal_direction/data/manifest.json').read_text())
 qt=AutoTokenizer.from_pretrained(old['model'],revision=old['revision'],cache_dir=CACHE,local_files_only=True)
 repo,rev=SPECS['gla'];tok=AutoTokenizer.from_pretrained(repo,revision=rev,cache_dir=CACHE,local_files_only=True)
 gd=AutoTokenizer.from_pretrained(SPECS['gdn'][0],revision=SPECS['gdn'][1],cache_dir=CACHE,local_files_only=True)
 probe='The quick brown fox jumps over the lazy dog.'
 assert tok(probe)['input_ids']==gd(probe)['input_ids']
 assert tok.get_vocab()==gd.get_vocab()
 records=[];seen=set();seen_tokens=set()
 for split,count,length in [('train',1024,512),('validation',32,512),('confirm_wiki',64,512),('confirm_long',16,2048)]:
  chosen=[]
  for r in old['records']:
   if r['split']!=split:continue
   text=qt.decode(r['input_ids'],skip_special_tokens=True)
   ids=tok(text,add_special_tokens=False)['input_ids']
   if len(ids)<length:continue
   ids=ids[:length];th=r['text_sha256'];ih=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
   if th in seen or ih in seen_tokens:continue
   seen.add(th);seen_tokens.add(ih)
   chosen.append(dict(split=split,index=r['index'],text_sha256=th,token_sha256=ih,input_ids=ids))
   if len(chosen)==count:break
  assert len(chosen)==count,(split,len(chosen));records+=chosen
 save(path,dict(tokenizer=repo,tokenizer_revision=rev,tokenizer_vocabulary_matches_gdn=True,records=records,
  counts={s:sum(r['split']==s for r in records) for s in ['train','validation','confirm_wiki','confirm_long']},
  scope='Retokenized existing disjoint document splits; 512-token single-pass train, 2048-token long test; not newly blind documents.'))

if __name__=='__main__':
 torch.set_num_threads(4)
 if sys.argv[1]=='data':manifest_prepare();print('data prepared',flush=True)
 else:
  family=sys.argv[1];model=model_load(family);docs=data_load()['records']
  norms=collect_norm(model,family,[r for r in docs if r['split']=='train'])
  print(family,'ready',flush=True)
