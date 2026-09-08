"""Actual long-context precision, scale invariance and window cache checks."""
from common import *
from operators import LayerFeatures,prefill
from runtime import Replacement,cache_for
from hybrid import Hybrid
from swa import Sliding
from transformers import AutoModelForCausalLM
from transformers.models.qwen2 import modeling_qwen2
from types import SimpleNamespace

@torch.inference_mode()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 manifest=json.loads((P/'data/manifest.json').read_text());records=[r for r in manifest['records'] if r['split']=='confirm_long'];original=modeling_qwen2.ALL_ATTENTION_FUNCTIONS['sdpa'];capture={}
 def hook(mod,q,k,v,mask,**kw):
  if mod.layer_idx in [14,27]:capture[mod.layer_idx]=(q,k,v)
  return original(mod,q,k,v,mask,**kw)
 modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',hook);model=AutoModelForCausalLM.from_pretrained(manifest['model'],revision=manifest['revision'],cache_dir='/root/autodl-tmp/hf-cache',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
 ids=torch.tensor([records[0]['input_ids']],device='cuda');model.model(input_ids=ids,use_cache=False);rows=[]
 for name in ['split_01_s11','exp_kl_s11','softplus_kl_s11','calibrated_s11','gate_s11','favor_s11','hedgehog_s11','learned_prf_s11']:
  for l in [14,27]:
   q,k,v=[x[0] for x in capture[l]];f=LayerFeatures(name,l,optimized=True);lq,lk=f.logs(q,k);y,den,state=prefill(lq,lk,v.float().repeat_interleave(6,0));f64=LayerFeatures(name,l,torch.float64,False);qi=torch.arange(0,8192,128,device='cuda');lp=f64.net.log_matrix(q[:,qi].double(),k.double().repeat_interleave(6,0));mask=torch.arange(8192,device='cuda')[None]<=qi[:,None];ref=lp.masked_fill(~mask[None],-torch.inf).softmax(-1)@v.double().repeat_interleave(6,0);err=float((y[:,qi].double()-ref).norm()/ref.norm());zeros=int((den<=0).sum());assert err<.002 and zeros==0,(name,l,err,zeros)
   aq,bk=f.logs(q[:,:1],k[:,:1]);ar=f.net.log_feature(q[:,:1].float(),'q');br=f.net.log_feature(k[:,:1].float().repeat_interleave(6,0),'k');ferr=max(float((aq-ar).abs().max()),float((bk-br).abs().max()));assert ferr<1e-3,(name,l,ferr)
   rows.append(dict(name=name,layer=l,relative_output_error_fp32_vs_explicit_fp64=err,zero_denominators=zeros,min_denominator=float(den.min()),fused_feature_max_log_error=ferr));del f,f64
 # SWA448+4 must select exactly its dense mask and have identical cached rule.
 q,k,v=capture[14];q=q[:,:,:600];k=k[:,:,:600];v=v[:,:,:600];mod=SimpleNamespace(layer_idx=14);rt=Sliding(None);full=rt(mod,q,k,v,None)[0];ix=torch.arange(600,device='cuda');mask=(ix[None]<=ix[:,None])&((ix[None]<4)|(ix[None]>ix[:,None]-448));ref=F.scaled_dot_product_attention(q[0],k[0].repeat_interleave(6,0),v[0].repeat_interleave(6,0),attn_mask=mask[None])[None].transpose(1,2);err=float((full.float()-ref.float()).norm()/ref.float().norm());assert err<.006
 rt.reset();parts=[rt(mod,q[:,:,:580],k[:,:,:580],v[:,:,:580],None)[0]]
 for t in range(580,600):parts.append(rt(mod,q[:,:,t:t+1],k[:,:,t:t+1],v[:,:,t:t+1],None)[0])
 cached=torch.cat(parts,1);cerr=float((cached.float()-ref.float()).norm()/ref.float().norm());assert cerr<.006
 # Pure attention directions exactly preserved by global/query calibration.
 ds=data('validation');q=ds['q'][0].cuda().double();k=ds['k'][0].cuda().double()[KI.cuda()];base,_=load('exp_kl_s11');a=base.log_matrix(q,k).softmax(-1);gauge=[]
 for name in ['calibrated_s11','gate_s11','global_bal_s11','global_raw_s11']:
  net,_=load(name);b=net.log_matrix(q,k).softmax(-1);e=float((a-b).abs().max());assert e<1e-11;gauge.append(dict(name=name,max_probability_difference=e))
 bounds=[]
 for l in [14,27]:
  layer=model.model.layers[l];gamma=layer.input_layernorm.weight.double();dh=128;dm=len(gamma);att=layer.self_attn
  for head in range(12):
   wq=att.q_proj.weight[head*dh:(head+1)*dh].double()*gamma;wk=att.k_proj.weight[(head//6)*dh:(head//6+1)*dh].double()*gamma;bq=att.q_proj.bias[head*dh:(head+1)*dh].double();bk=att.k_proj.bias[(head//6)*dh:(head//6+1)*dh].double();bqmax=math.sqrt(dm)*wq.norm()+bq.norm();bkmax=math.sqrt(dm)*wk.norm()+bk.norm();bounds.append(dict(layer=l,head=head,conservative_log_kernel_cap=float(bqmax*bkmax/math.sqrt(dh))))
 save(P/'checks/architecture_moment_bound.json',dict(theorem='RMSNorm bounds projected Q/K norms; default orthogonal RoPE preserves them. Hence every raw-kernel moment exists for the fixed model.',numeric_bounds=bounds,scope='Conservative Frobenius replacement for operator norm, not a tight support bound; finite moments do not validate an unbounded Gaussian approximation or a useful sample concentration rate.'))
 save(P/'checks/long_precision_and_window.json',dict(passed=True,long_context=rows,swa=dict(dense_mask_relative_error=err,prefill_decode_relative_error=cerr),gauge_invariance=gauge));modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

if __name__=='__main__':main()
