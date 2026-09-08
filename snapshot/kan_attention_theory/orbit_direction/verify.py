import sys
from pathlib import Path
O=Path(__file__).resolve().parent;C=O.parent/'causal_direction';sys.path.insert(0,str(C))
from common import *
from operators import LayerFeatures,prefill
from runtime import Replacement,cache_for
from hybrid import Hybrid,hybrid_storage
from transformers import AutoModelForCausalLM
from transformers.models.qwen2 import modeling_qwen2

@torch.inference_mode()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;manifest=json.loads((O/'data/manifest.json').read_text());record=next(r for r in manifest['records'] if r['split']=='orbit_long');original=modeling_qwen2.ALL_ATTENTION_FUNCTIONS['sdpa'];capture={}
 def hook(mod,q,k,v,mask,**kw):
  if mod.layer_idx in [14,27]:capture[mod.layer_idx]=(q,k,v)
  return original(mod,q,k,v,mask,**kw)
 modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',hook);model=AutoModelForCausalLM.from_pretrained(manifest['model'],revision=manifest['revision'],cache_dir='/root/autodl-tmp/hf-cache',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();ids=torch.tensor([record['input_ids']],device='cuda');model.model(input_ids=ids,use_cache=False);rows=[]
 for name in [f'orbit_{k}_s{s}' for k in ['split_01','split_kl'] for s in [11,29,47]]:
  for l in [14,27]:
   q,k,v=[x[0] for x in capture[l]];f=LayerFeatures(name,l);lq,lk=f.logs(q,k);y,den,st=prefill(lq,lk,v.float().repeat_interleave(6,0));f64=LayerFeatures(name,l,torch.float64,False);qi=torch.arange(0,8192,128,device='cuda');lp=f64.net.log_matrix(q[:,qi].double(),k.double().repeat_interleave(6,0));mask=torch.arange(8192,device='cuda')[None]<=qi[:,None];ref=lp.masked_fill(~mask[None],-torch.inf).softmax(-1)@v.double().repeat_interleave(6,0);err=float((y[:,qi].double()-ref).norm()/ref.norm());assert err<.002 and (den>0).all();a,b=f.logs(q[:,:1],k[:,:1]);ar=f.net.log_feature(q[:,:1].float(),'q');br=f.net.log_feature(k[:,:1].float().repeat_interleave(6,0),'k');ferr=max(float((a-ar).abs().max()),float((b-br).abs().max()));assert ferr<.001
   rows.append(dict(name=name,layer=l,fp32_vs_fp64_relative_output_error=err,fused_max_log_feature_error=ferr,zero_denominators=int((den<=0).sum())));del f,f64
 checks=[]
 for cls in [Replacement,Hybrid]:
  for name in ['orbit_split_01_s11','orbit_split_kl_s11']:
   rt=cls(original,name);rt.collect=False;modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',rt);rt.reset();full=model(input_ids=ids[:,:1024],use_cache=False,logits_to_keep=1).logits.float();rt.reset();cache=cache_for(model,True);out=model(input_ids=ids[:,:1000],past_key_values=cache,use_cache=True,logits_to_keep=1)
   for t in range(1000,1024):out=model(input_ids=ids[:,t:t+1],past_key_values=cache,use_cache=True,logits_to_keep=1)
   err=float((out.logits.float()-full).norm()/full.norm());assert err<.03;sz=hybrid_storage(cache,rt);assert sz['state_bytes']==798720 and cache.layers[14].keys.untyped_storage().nbytes()==cache.layers[27].keys.untyped_storage().nbytes()==0;checks.append(dict(name=name,mode=cls.__name__,logit_relative_error=err,top1_equal=bool((out.logits.argmax(-1)==full.argmax(-1)).all()),**sz))
 modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',original);save(O/'checks/runtime.json',dict(passed=True,precision_checks=rows,model_cache_checks=checks));del model,capture,rt,cache,out
 fresh=torch.load(O/'data/orbit_wiki_000.pt',weights_only=True);freq=1e6**(-torch.arange(0,128,2,device='cuda',dtype=torch.float64)/128);stress=[];max_shift_error=0.
 def rotate(x,delta):
  a,b=x[...,:64],x[...,64:];phase=delta*freq;c=phase.cos();s=phase.sin();return torch.cat([a*c-b*s,b*c+a*s],-1)
 for name in [f'{k}_s{s}' for k in ['split_01','split_kl','orbit_split_01','orbit_split_kl'] for s in [11,29,47]]:
  net,meta=load(name);scale=torch.tensor(meta['log_scale'],device='cuda',dtype=torch.float64)
  for delta in [0,1024,8192]:
   stats=[]
   for i in range(16):
    q0=fresh['q'][i].cuda().double();k0=fresh['k'][i].cuda().double()[KI.cuda()];q=rotate(q0,delta);k=rotate(k0,delta);lt=q0@k0.transpose(-1,-2)/math.sqrt(D)-scale[:,None,None];shifted=q@k.transpose(-1,-2)/math.sqrt(D)-scale[:,None,None];err=float((lt-shifted).abs().max());max_shift_error=max(max_shift_error,err);assert err<1e-10
    mask=(torch.arange(1024,device='cuda')[None]<=fresh['query_positions'][i].cuda()[:,None])[None];loss,kl,mass=rows_loss(net.log_matrix(q,k),lt,mask,1.);stats.append(dict(ordinal=i,kl=kl.mean(-1).tolist(),mass=mass.mean(-1).tolist()))
   stress.append(dict(name=name,offset=delta,mean_kl=sum(sum(r['kl']) for r in stats)/(16*24),mean_mass=sum(sum(r['mass']) for r in stats)/(16*24),documents=stats))
 save(O/'results/rotation_confirmation.json',dict(results=stress,max_raw_logit_shift_error=max_shift_error,scope='First16 newly heldout Wiki documents, three training seeds, fixed offsets0/1024/8192; same raw target verified under each common rotation. No fitting.'))
 # Verify training budgets, exact pair/order matching, and fresh document hashes.
 metas={}
 for f in (O/'fits').glob('*.pt'):metas[f.stem]=torch.load(f,weights_only=True,map_location='cpu')['metadata']
 assert len(metas)==6
 for seed in [11,29,47]:
  a=metas[f'orbit_split_01_s{seed}'];b=metas[f'orbit_split_kl_s{seed}'];old=json.loads((C/'fits'/f'split_01_s{seed}.json').read_text())['metadata']
  for key in ['query_sha256','order_sha256','training_pairs_per_head','parameters_per_head']:
   assert a[key]==b[key]==old[key]
  assert a['offsets_sha256']==b['offsets_sha256'] and a['epochs']==b['epochs']==1
 # The collector excluded all historical manifests; retain a direct current
 # train/validation/confirmation disjointness audit too.
 prev=json.loads((C/'data/manifest.json').read_text())['records']
 for key in ['text_sha256','token_sha256']:
  assert not {r[key] for r in manifest['records']}&{r[key] for r in prev}
 assert len(list((O/'results').glob('kernel_*.json')))==36
 assert len(list((O/'results').glob('ppl_*.json')))==64
 save(O/'checks/final_audit.json',dict(passed=True,trained_fits=6,kernel_evaluations=36,full_model_evaluations=64,training_pairs_per_head=134360517,parameters_per_head=73856,fresh_wiki_documents=64,fresh_long_documents=16,scope='Same24heads, two complete frozen-model layers. Same-budget augmentation and matched KL control. Corpus generalization remains limited to new Wiki documents and length; not a second model or broad task evaluation.'))

if __name__=='__main__':main()
