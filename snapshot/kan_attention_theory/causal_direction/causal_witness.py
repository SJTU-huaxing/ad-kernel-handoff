"""Posthoc theorem diagnostic, fixed candidates; exact dyadic causal rectangles."""
from common import *
from transformers import AutoModelForCausalLM
from transformers.models.qwen2 import modeling_qwen2

def rectangles(n,left=0):
 if n<=1:return []
 half=n//2;mid=left+half
 return [(mid,left+n,left,mid)]+rectangles(half,left)+rectangles(n-half,mid)

@torch.inference_mode()
def bound(a,ranks=[16,32,64,128]):
 # a contains row-normalized causal probabilities; Q is uniform over positions.
 h,n,_=a.shape;out=torch.zeros(h,len(ranks),device=a.device,dtype=a.dtype)
 for q0,q1,k0,k1 in rectangles(n):
  if min(q1-q0,k1-k0)<=min(ranks):continue
  sub=a[:,q0:q1,k0:k1]/n;mass=sub.sum((-1,-2));C=sub/mass.clamp_min(1e-300)[:,None,None]
  # LAPACK is much faster than consumer-GPU Jacobi for these small FP64 SVDs.
  sv=torch.linalg.svdvals(C.cpu()).to(a.device)
  for j,m in enumerate(ranks):
   if len(sv[0])>m:out[:,j]+=.5*mass*sv[:,m:].sum(-1).square()
 return out

@torch.inference_mode()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(891)
 n=128;m=8;truth=torch.randn(2,n,n,device='cuda',dtype=torch.float64);h=torch.rand(2,n,m,device='cuda',dtype=torch.float64)@torch.rand(2,m,n,device='cuda',dtype=torch.float64);mask=torch.ones(n,n,device='cuda',dtype=torch.bool).tril();a=truth.masked_fill(~mask,-torch.inf).softmax(-1);b=h.log().masked_fill(~mask,-torch.inf).softmax(-1);kl=(a*(a.clamp_min(1e-300).log()-b.clamp_min(1e-300).log())).sum((-1,-2))/n;lb=bound(a,[m])[:,0];assert (kl>=lb).all()
 cover=torch.eye(n,dtype=torch.int32)
 for q0,q1,k0,k1 in rectangles(n):cover[q0:q1,k0:k1]+=1
 assert torch.equal(cover,torch.ones(n,n,dtype=torch.int32).tril())
 # Exact rank-m predictions must have zero coarsened spectral tail within EACH
 # product rectangle, despite the full causal matrix typically having rank n.
 zeros=bound(b,[m]);assert float(zeros.max())<1e-22
 manifest=json.loads((P/'data/manifest.json').read_text());records=[r for r in manifest['records'] if r['split']=='confirm_wiki'];original=modeling_qwen2.ALL_ATTENTION_FUNCTIONS['sdpa'];capture={}
 def hook(module,q,k,v,mask,**kw):
  if module.layer_idx in [14,27]:capture[module.layer_idx]=(q,k)
  return original(module,q,k,v,mask,**kw)
 modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',hook);model=AutoModelForCausalLM.from_pretrained(manifest['model'],revision=manifest['revision'],cache_dir='/root/autodl-tmp/hf-cache',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
 nets={name:load(name)[0] for name in ['split_01_s11','exp_kl_s11','calibrated_s11']};rows=[];start=time.perf_counter()
 model.model(input_ids=torch.tensor([records[0]['input_ids']],device='cuda'),use_cache=False)
 for i,r in enumerate(records):
  model.model(input_ids=torch.tensor([r['input_ids']],device='cuda'),use_cache=False);q=torch.cat([capture[l][0][0,:,:512] for l in [14,27]]).double();k=torch.cat([capture[l][1][0,:,:512] for l in [14,27]]).double()[KI.cuda()]
  t=q@k.transpose(-1,-2)/math.sqrt(D);mask=torch.ones(512,512,device='cuda',dtype=torch.bool).tril()[None];t=t.masked_fill(~mask,-torch.inf);a=t.softmax(-1);lb=bound(a);risks={}
  for name,net in nets.items():
   p=net.log_matrix(q,k).masked_fill(~mask,-torch.inf);kl=(a*(t-t.logsumexp(-1,keepdim=True)-p+p.logsumexp(-1,keepdim=True)).masked_fill(~mask,0)).sum(-1).mean(-1);assert (kl+1e-10>=lb[:,2]).all();risks[name]=kl.tolist()
  rows.append(dict(ordinal=i,bounds=lb.tolist(),kl=risks))
  if (i+1)%16==0:print(json.dumps(dict(event='causal_witness',documents=i+1,seconds=time.perf_counter()-start)),flush=True)
 summary=torch.tensor([r['bounds'] for r in rows]).double().mean(0);correction=.5*math.sqrt(math.log(20)/(2*len(rows)))
 save(P/'results/causal_witness.json',dict(ranks=[16,32,64,128],documents=rows,summary=summary.tolist(),iid_document_95_hoeffding_lower=(summary-correction).clamp_min(0).tolist(),confidence_correction=correction,scope='Diagnostic added after inspecting first confirmation kernel metrics; no fitting or candidate selection. Exact prefix512 causal distribution in each of128 heldout Wiki documents. Universal conditional-context inequality, empirical average estimates expectation over documents. Unknown-document lower CI assumes iid documents. It does not establish an8192-token lower certificate or tightness.',checks=dict(dyadic_cover_exact=True,rank_m_prediction_zero_bound=float(zeros.max()),random_kl=kl.tolist() if False else None),seconds=time.perf_counter()-start));modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

if __name__=='__main__':main()
