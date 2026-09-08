"""Audit the Gaussian moment condition on all cached training Q and K."""
from common import *

@torch.inference_mode()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 sums={s:torch.zeros(h,D,device='cuda',dtype=torch.float64) for s,h in [('q',24),('k',4)]};products={s:torch.zeros(h,D,D,device='cuda',dtype=torch.float64) for s,h in [('q',24),('k',4)]};counts=dict(q=0,k=0)
 for f in sorted((P/'data').glob('train_*.pt')):
  box=torch.load(f,weights_only=True)
  for s in ['q','k']:
   x=box[s].permute(1,0,2,3).flatten(1,2).cuda().double();sums[s]+=x.sum(1);products[s]+=x.transpose(-1,-2)@x;counts[s]+=x.shape[1]
 cov={s:(products[s]-sums[s][:,:,None]*sums[s][:,None,:]/counts[s])/(counts[s]-1) for s in ['q','k']};e,u=torch.linalg.eigh(cov['q']);root=(u*e.clamp_min(0).sqrt()[:,None])@u.transpose(-1,-2);aa=root@cov['k'][KI.cuda()]@root/D;aa=(aa+aa.transpose(-1,-2))/2;singular=torch.linalg.eigvalsh(aa).clamp_min(0).sqrt();valid=singular[:,-1]<.5
 target=P/'checks/gaussian_moment_domain.json'
 if target.exists():save(P/'checks/gaussian_moment_domain_initial_subsample.json',json.loads(target.read_text()))
 save(target,dict(heads=HEADS,maximum_a=singular[:,-1].tolist(),hilbert_schmidt_valid=valid.tolist(),valid_heads=int(valid.sum()),total_heads=H,q_vectors_per_head=counts['q'],k_vectors_per_group=counts['k'],scope='Exact covariance of all cached training Q (64 uniformly sampled distinct positions/document) and all1024 K/document. No position-strided subsample. This checks Gaussian-surrogate domain, not Gaussianity; actual bounded RMSNorm model has finite raw-kernel moments.'));print(json.dumps(dict(valid_heads=int(valid.sum()),counts=counts,amin=float(singular[:,-1].min()),amax=float(singular[:,-1].max()))),flush=True)

if __name__=='__main__':main()
