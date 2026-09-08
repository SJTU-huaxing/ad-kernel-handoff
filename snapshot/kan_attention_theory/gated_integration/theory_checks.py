"""Nontrivial algebra/stability checks on real native and Qwen feature data."""
from native import *

def rms(x,eps=0):return x/(x.square().mean(-1,keepdim=True)+eps).sqrt().clamp_min(1e-100)

def delta(q,k,v,alpha,beta,write=None):
 h,t,m=k.shape;dv=v.shape[-1];s=torch.zeros(h,m,dv,dtype=k.dtype,device=k.device);out=[];norms=[]
 if write is None:write=beta
 for j in range(t):
  s=s*alpha[:,j,None,None]
  old=(k[:,j,:,None]*s).sum(1)
  s=s+k[:,j,:,None]*(write[:,j,None]*v[:,j]-beta[:,j,None]*old)[:,None,:]
  out.append((q[:,j,:,None]*s).sum(1));norms.append(s.norm(dim=(-1,-2)))
 return torch.stack(out,1),torch.stack(norms,1),s

def main():
 torch.set_num_threads(4);torch.manual_seed(713)
 # Exact path-kernel expansion, scalar-gated I homogeneity, and scale cancellation.
 h,t,m,dv=2,19,5,7
 q=torch.rand(h,t,m,dtype=torch.float64);k=torch.rand_like(q);v=torch.randn(h,t,dv,dtype=torch.float64)
 gates=.7+.3*torch.rand_like(q)
 s=torch.zeros(h,m,dv,dtype=torch.float64);seq=[];explicit=[]
 for i in range(t):
  s=gates[:,i,:,None]*s+k[:,i,:,None]*v[:,i,None,:]
  seq.append((q[:,i,:,None]*s).sum(1))
  row=[]
  for j in range(i+1):
   path=gates[:,j+1:i+1].prod(1)
   row.append((q[:,i]*path*k[:,j]).sum(-1))
  explicit.append((torch.stack(row,1)[...,None]*v[:,:i+1]).sum(1))
 gla_error=float((torch.stack(seq,1)-torch.stack(explicit,1)).abs().max());assert gla_error<1e-12
 a=torch.exp(torch.randn(101,dtype=torch.float64));b=torch.exp(torch.randn_like(a));w=torch.rand_like(a)
 di=lambda x,y:x*torch.log(x/y)-x+y
 homog=float((di(w*a,w*b)-w*di(a,b)).abs().max());assert homog<1e-12
 # Positive q,k need not yield nonnegative delta effective attention.
 kk=torch.tensor([[[1.,0.],[2**-.5,2**-.5]]],dtype=torch.float64)
 qq=torch.tensor([[[1.,0.],[0.,1.]]],dtype=torch.float64)
 vv=torch.eye(2,dtype=torch.float64)[None]
 yy,_,_=delta(qq,kk,vv,torch.ones(1,2,dtype=torch.float64),torch.ones(1,2,dtype=torch.float64))
 assert yy[0,1,0]<0
 rows=[]
 for family in ['gla','gdn']:
  path=P/'data'/f'{family}_operator_samples.pt'
  if not path.exists():continue
  ds=torch.load(path,weights_only=True);norms=torch.load(P/'data'/f'{family}_norm.pt',weights_only=True)
  for l in FAMILY_LAYERS[family]:
   sample=ds[0][l];q0=sample['q'].double();k0=sample['k'].double();v0=sample['v'].double()
   # FeatureMaps casts to FP32 intentionally for deployment, here test raw real vectors in FP64.
   qn=F.normalize(q0,dim=-1);kn=F.normalize(k0,dim=-1)
   amp=torch.exp(torch.linspace(-8,8,q0.shape[1],dtype=torch.float64))[None,:,None,None]
   norm_error=float((F.normalize(q0*amp,dim=-1)-qn).abs().max())
   key_norm_error=float((F.normalize(k0/amp,dim=-1)-kn).abs().max())
   assert norm_error<1e-12 and key_norm_error<1e-12
   b=.5;scaled_norm2=(kn*amp).square().sum(-1)
   unstable=float((b*scaled_norm2>2).double().mean())
   rows.append(dict(family=family,layer=l,l2_query_amplitude_cancellation_max_abs=norm_error,
    l2_key_amplitude_cancellation_max_abs=key_norm_error,
    controlled_amplitude_sweep_log_range=[-8,8],unstable_fraction_if_skipping_l2_beta_half=unstable))
 # An actual counterexample: decay alone does not fix an expansive delta eigenvalue.
 unstable_gain=.99*abs(1-.5*3**2);assert unstable_gain>1
 qq=torch.ones(1,2,1,dtype=torch.float64);kk=torch.ones_like(qq)
 vv=torch.tensor([[[1.],[0.]]],dtype=torch.float64);aa=torch.ones(1,2,dtype=torch.float64);bb=.5*aa
 base,_,_=delta(qq,kk,vv,aa,bb);scaled,_,_=delta(qq/3,kk*3,vv,aa,bb)
 assert torch.allclose(qq@kk.transpose(-1,-2),(qq/3)@(kk*3).transpose(-1,-2))
 assert abs(float(base[0,1,0])-.25)<1e-12 and abs(float(scaled[0,1,0])+1.75)<1e-12
 # Stronger example: strictly positive UNIT vectors and identical complete QK matrix.
 # Thus the obstruction survives native L2 normalization, without unstable scales.
 qu=F.normalize(torch.ones(1,2,3,dtype=torch.float64),dim=-1)
 ku1=F.normalize(torch.tensor([[[2.,1.,1.],[2.,1.,1.]]],dtype=torch.float64),dim=-1)
 ku2=F.normalize(torch.tensor([[[2.,1.,1.],[1.,2.,1.]]],dtype=torch.float64),dim=-1)
 cross_error=float((qu@ku1.transpose(-1,-2)-qu@ku2.transpose(-1,-2)).abs().max())
 out1,_,_=delta(qu,ku1,vv,aa,aa);out2,_,_=delta(qu,ku2,vv,aa,aa)
 assert cross_error<1e-12 and abs(float(out1[0,1,0]))<1e-12
 assert abs(float(out2[0,1,0])-4/(6*math.sqrt(18)))<1e-12
 save(P/'checks/theory.json',dict(passed=True,gla_path_expansion_max_abs=gla_error,
  scalar_gated_I_homogeneity_max_abs=homog,positive_delta_negative_weight=float(yy[0,1,0]),
  skipped_l2_counterexample_gain=unstable_gain,real_native_vector_checks=rows,
  identical_static_kernel_different_delta_outputs=dict(original=base.flatten().tolist(),rescaled=scaled.flatten().tolist()),
  identical_unit_positive_cross_kernel_different_delta_outputs=dict(
   cross_kernel_max_difference=cross_error,keys_identical_outputs=out1.flatten().tolist(),
   keys_permuted_outputs=out2.flatten().tolist(),key_key_second_dot=5/6),
  scope='Algebra and controlled perturbations of real vectors; not a new accuracy or population-optimality result.'))
 print('theory checks passed',flush=True)

if __name__=='__main__':main()
