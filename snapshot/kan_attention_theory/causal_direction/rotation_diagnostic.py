"""Frozen-feature common-RoPE-shift stress test; raw target exactly unchanged."""
from common import *
from transformers import AutoConfig

@torch.inference_mode()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 manifest=json.loads((P/'data/manifest.json').read_text());cfg=AutoConfig.from_pretrained(manifest['model'],revision=manifest['revision'],cache_dir='/root/autodl-tmp/hf-cache',local_files_only=True)
 theta=(getattr(cfg,'rope_parameters',None) or {}).get('rope_theta',getattr(cfg,'rope_theta',1000000.))
 freq=theta**(-torch.arange(0,D,2,device='cuda',dtype=torch.float64)/D)
 def rotate(x,delta):
  phase=delta*freq;c=phase.cos();s=phase.sin();a,b=x[...,:D//2],x[...,D//2:];return torch.cat([a*c-b*s,b*c+a*s],-1)
 ds=data('confirm_wiki');rows=[];maxerr=0.
 # Separate audit of the Gaussian surrogate's Hilbert-Schmidt domain.
 tr=data('train');cq=tr['q'][:,:,::16].permute(1,0,2,3).flatten(1,2).cuda().double();ck=tr['k'][:,:,::256].permute(1,0,2,3).flatten(1,2).cuda().double();del tr
 cq-=cq.mean(1,keepdim=True);ck-=ck.mean(1,keepdim=True);sq=cq.transpose(-1,-2)@cq/(cq.shape[1]-1);sk=ck.transpose(-1,-2)@ck/(ck.shape[1]-1);eig,u=torch.linalg.eigh(sq);root=(u*eig.clamp_min(0).sqrt()[:,None])@u.transpose(-1,-2);aa=root@sk[KI.cuda()]@root/D;aa=(aa+aa.transpose(-1,-2))/2;singular=torch.linalg.eigvalsh(aa).clamp_min(0).sqrt();valid=singular[:,-1]<.5
 save(P/'checks/gaussian_moment_domain.json',dict(heads=HEADS,maximum_a=singular[:,-1].tolist(),hilbert_schmidt_valid=valid.tolist(),valid_heads=int(valid.sum()),total_heads=H,sampled_q_vectors_per_head=cq.shape[1],sampled_k_vectors_per_group=ck.shape[1],scope='Covariances from four train-only Q/K per each4096 documents. Domain check of Gaussian surrogate, not proof of Gaussianity or of true-kernel moment divergence. Actual RMSNorm model is bounded.'))
 del cq,ck,sq,sk,root,aa,singular,eig,u
 for name in [f'{k}_s11' for k in ['split_1','split_01','split_kl','exp_kl','softplus_kl','calibrated','gate','favor']]:
  net,meta=load(name);scale=torch.tensor(meta['log_scale'],device='cuda',dtype=torch.float64)
  for delta in [0,1024,4096,8192,32768]:
   rr=[]
   for i in range(16):
    q0=ds['q'][i].cuda().double();k0=ds['k'][i].cuda().double()[KI.cuda()];q=rotate(q0,delta);k=rotate(k0,delta);truth=q0@k0.transpose(-1,-2)/math.sqrt(D)-scale[:,None,None];shifted=q@k.transpose(-1,-2)/math.sqrt(D)-scale[:,None,None];err=float((truth-shifted).abs().max());maxerr=max(maxerr,err);assert err<1e-10
    mask=(torch.arange(1024,device='cuda')[None]<=ds['query_positions'][i].cuda()[:,None])[None];lp=net.log_matrix(q,k);loss,kl,mass=rows_loss(lp,truth,mask,1.)
    rr.append(dict(ordinal=i,kl=kl.mean(-1).tolist(),mass=mass.mean(-1).tolist(),log_kernel_mae=((truth-lp).abs()*mask).sum((-1,-2)).div(mask.sum()).tolist()))
   summ={k:torch.tensor([r[k] for r in rr],dtype=torch.float64).mean(0).tolist() for k in rr[0] if k!='ordinal'};rows.append(dict(name=name,common_offset=delta,summary=summ,documents=rr))
  print(json.dumps(dict(event='rotation',name=name)),flush=True)
 save(P/'results/rotation_diagnostic.json',dict(rope_theta=theta,max_raw_logit_invariance_error=maxerr,results=rows,scope='Posthoc structural diagnostic: first16 fixed Wiki confirmation docs, seed11 only, no fitting. Simultaneous orthogonal RoPE rotations preserve every raw target kernel value, causal mask and exact oracle spectrum. They isolate feature sensitivity, but do not establish that this is the sole cause of genuine long-context failure.'))

if __name__=='__main__':main()
