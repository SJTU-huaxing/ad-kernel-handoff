from common import *
from operators import LayerFeatures,prefill,step
from hybrid import Hybrid,hybrid_storage
from runtime import cache_for,Replacement
from transformers import AutoModelForCausalLM
from transformers.models.qwen2 import modeling_qwen2
from types import SimpleNamespace

@torch.inference_mode()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(791)
 rows=[]
 # Mathematical recurrence: independently explicit positive matrix, FP64.
 lq=torch.randn(12,193,64,device='cuda',dtype=torch.float64)*2;lk=torch.randn_like(lq)*2;v=torch.randn(2,193,128,device='cuda',dtype=torch.float64);vg=v.repeat_interleave(6,0)
 logits=torch.logsumexp(lq[:,:,None,:]+lk[:,None,:,:],-1).tril() # construct below with correct -inf mask
 logits=torch.logsumexp(lq[:,:,None,:]+lk[:,None,:,:],-1).masked_fill(~torch.ones(193,193,device='cuda',dtype=torch.bool).tril(),-torch.inf)
 exact=logits.softmax(-1)@vg;y,den,st=prefill(lq,lk,vg);err=float((y-exact).norm()/exact.norm());assert err<1e-12;rows.append(dict(check='FP64 explicit kernel vs chunk scan',relative_error=err))
 yp,_,st=prefill(lq[:,:160],lk[:,:160],vg[:,:160]);tail=[]
 for t in range(160,193):z,_=step(lq[:,t:t+1],lk[:,t:t+1],v[:,t:t+1],st,False);tail.append(z)
 yy=torch.cat([yp]+tail,1);err=float((yy-exact).norm()/exact.norm());assert err<1e-12;rows.append(dict(check='FP64 recurrence',relative_error=err))
 y,_,st=prefill(lq.float()[:,:160],lk.float()[:,:160],vg.float()[:,:160]);tail=[]
 for t in range(160,193):z,_=step(lq.float()[:,t:t+1],lk.float()[:,t:t+1],v.to(torch.bfloat16)[:,t:t+1],st,True);tail.append(z)
 # Reference uses the same BF16 V for the decode segment.
 vv=vg.clone();vv[:,160:]=vv[:,160:].bfloat16().double();ref=logits.softmax(-1)@vv
 err=float((torch.cat([y]+tail,1).double()-ref).norm()/ref.norm());assert err<1e-5;rows.append(dict(check='FP32 Triton state update with BF16 values',relative_error=err))
 ds=data('validation');q=ds['q'][0,:12].cuda();k=ds['k'][0,:2].cuda();vv=ds['v'][0,:2].cuda()
 for name in ['split_01_s11','exp_kl_s11','softplus_kl_s11','calibrated_s11','gate_s11','favor_s11']:
  f=LayerFeatures(name,14);a,b=f.logs(q[:,:1],k[:,:1]);ar=f.net.log_feature(q[:,:1].float(),'q');br=f.net.log_feature(k[:,:1].float().repeat_interleave(6,0),'k')
  err=max(float((a-ar).abs().max()),float((b-br).abs().max()));assert err<1e-3,(name,err);rows.append(dict(check='fused log features',name=name,max_absolute_error=err))
  # The same real post-RoPE Q vectors are reused as a synthetic sequence here;
  # exact/scan comparison, not a quality measurement on mismatched positions.
  qseq=q; kseq=k[:,:64];vseq=vv[:,:64];mod=SimpleNamespace(layer_idx=14)
  f.net=None;del f
  for cls in [Replacement,Hybrid]:
   rt=cls(None,name);rt.collect=False
   full=rt(mod,qseq[None],kseq[None],vseq[None],None)[0]
   rt.reset();parts=[rt(mod,qseq[None,:,:48],kseq[None,:,:48],vseq[None,:,:48],None)[0]]
   for t in range(48,64):parts.append(rt(mod,qseq[None,:,t:t+1],kseq[None,:,t:t+1],vseq[None,:,t:t+1],None)[0])
   # <=64 hybrid is fully exact; test remote mixture with 97 tokens below.
   cached=torch.cat(parts,1);err=float((full.float()-cached.float()).norm()/full.float().norm());assert err<.006,(name,cls.__name__,err)
   rows.append(dict(check='real feature prefill/cache',name=name,runtime=cls.__name__,relative_error=err));del rt
  rt=Hybrid(None,name);rt.collect=False;q97=q[:,torch.arange(97,device='cuda')%64];k97=k[:,:97];v97=vv[:,:97]
  out=rt(mod,q97[None],k97[None],v97[None],None)[0][0].transpose(0,1).float()
  net=rt.maps[14].net;lp=net.log_matrix(q97.float(),k97.float().repeat_interleave(6,0));scale=torch.tensor(rt.maps[14].meta['log_scale'][:12],device='cuda')
  exactlog=q97.float()@k97.float().repeat_interleave(6,0).transpose(-1,-2)/math.sqrt(D)-scale[:,None,None]
  ix=torch.arange(97,device='cuda');local=(ix[None]>ix[:,None]-64)&(ix[None]<=ix[:,None]);causal=ix[None]<=ix[:,None]
  matrix=torch.where(local[None],exactlog,lp).masked_fill(~causal[None],-torch.inf);ref=matrix.softmax(-1)@v97.float().repeat_interleave(6,0)
  err=float((out-ref).norm()/ref.norm());assert err<.006,(name,err);rows.append(dict(check='hybrid vs explicit raw mixed kernel',name=name,relative_error=err))
  rt.reset();parts=[rt(mod,q97[None,:,:80],k97[None,:,:80],v97[None,:,:80],None)[0]]
  for t in range(80,97):parts.append(rt(mod,q97[None,:,t:t+1],k97[None,:,t:t+1],v97[None,:,t:t+1],None)[0])
  cached=torch.cat(parts,1)[0].transpose(0,1).float();err=float((cached-ref).norm()/ref.norm());assert err<.006,(name,err);rows.append(dict(check='hybrid prefill+Triton decode vs explicit',name=name,relative_error=err));del rt
 # An algebraic hybrid mass KL identity independent of the implementation.
 t=torch.randn(31,129,dtype=torch.float64);p=torch.randn_like(t);p[:,-64:]=t[:,-64:];a=t.softmax(-1);b=p.softmax(-1);kl=(a*(a.log()-b.log())).sum(-1)
 alpha=a[:,-64:].sum(-1);beta=b[:,-64:].sum(-1);ber=alpha*(alpha/beta).log()+(1-alpha)*((1-alpha)/(1-beta)).log();ar=t[:,:-64].softmax(-1);br=p[:,:-64].softmax(-1);decomp=ber+(1-alpha)*(ar*(ar.log()-br.log())).sum(-1)
 assert torch.allclose(kl,decomp,atol=1e-12);rows.append(dict(check='exact-local / approximate-remote KL decomposition',max_error=float((kl-decomp).abs().max())))
 save(P/'checks/operators.json',dict(passed=True,checks=rows));print(json.dumps(dict(event='operators_passed',count=len(rows))),flush=True)
 # Full-model cache path checked separately from low-level algebra.
 m=json.loads((P/'data/manifest.json').read_text());original=modeling_qwen2.ALL_ATTENTION_FUNCTIONS['sdpa'];model=AutoModelForCausalLM.from_pretrained(m['model'],revision=m['revision'],cache_dir='/root/autodl-tmp/hf-cache',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();ids=ds['input_ids'][0:1,:256].cuda();checks=[]
 for cls,name in [(Replacement,'teacher'),(Replacement,'split_01_s11'),(Replacement,'exp_kl_s11'),(Hybrid,'calibrated_s11')]:
  rt=cls(original,name);modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',rt);rt.reset();full=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits.float()
  rt.reset();cache=cache_for(model,name!='teacher');out=model(input_ids=ids[:,:240],past_key_values=cache,use_cache=True,logits_to_keep=1)
  for t in range(240,256):out=model(input_ids=ids[:,t:t+1],past_key_values=cache,use_cache=True,logits_to_keep=1)
  err=float((full-out.logits.float()).norm()/full.norm());assert err<.03,(name,err);sz=hybrid_storage(cache,rt)
  if name!='teacher':assert cache.layers[14].keys.untyped_storage().nbytes()==cache.layers[27].keys.untyped_storage().nbytes()==0
  checks.append(dict(name=name,runtime=cls.__name__,relative_logit_error=err,top1_equal=bool((out.logits.argmax(-1)==full.argmax(-1)).all()),**sz));del rt,cache,out
 modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',original);save(P/'checks/model_cache.json',dict(passed=True,checks=checks,note='BF16 prefill/decode differ in accumulation paths; independent FP64 recurrence and FP32 operators verified above.'))

if __name__=='__main__':main()
