import native as n
from native import *
from theory_checks import delta

def main(family):
 torch.set_num_threads(4);patch_kernels()
 ds=torch.load(P/'data'/f'{family}_operator_samples.pt',weights_only=True)
 norms=torch.load(P/'data'/f'{family}_norm.pt',weights_only=True)
 rows=[]
 for l in FAMILY_LAYERS[family]:
  record=ds[0][l]
  q=record['q'][:,:128].cuda();k=record['k'][:,:128].cuda();v=record['v'][:,:128].cuda()
  kinds=['native']+KINDS[family]
  for kind in kinds:
   if kind=='native':qq,kk,vv=q,k,v
   else:
    net=FeatureMaps(family,kind,norms[str(l)],11+1000*l).cuda()
    with torch.no_grad():qq,kk,vv=net(q,k,v)
   a,b,c=[x[0].double().cpu().transpose(0,1) for x in [qq,kk,vv]]
   if family=='gla':
    g=record['g'][:,:128].cuda()
    with torch.no_grad():
     chunk,_=ORIGINAL[family,'chunk_gla'](q=qq,k=kk,v=vv,g=g,scale=1.,state_v_first=True)
     rec,_=ORIGINAL[family,'fused_recurrent_gla'](q=qq,k=kk,v=vv,gk=g,scale=1.,state_v_first=True)
    gg=g[0].double().cpu().transpose(0,1).exp()
    ss=torch.zeros(a.shape[0],a.shape[-1],c.shape[-1],dtype=torch.float64);ys=[]
    for t in range(a.shape[1]):
     ss=gg[:,t,:,None]*ss+b[:,t,:,None]*c[:,t,None,:]
     ys.append((a[:,t,:,None]*ss).sum(1))
    truth=torch.stack(ys,1).transpose(0,1)[None]
   else:
    raw=record['g'][:,:128].cuda();bb=record['beta'][:,:128].cuda()
    common=dict(q=qq,k=kk,v=vv,g=raw,beta=bb,A_log=record['A_log'].cuda(),dt_bias=record['dt_bias'].cuda(),
     scale=1.,use_qk_l2norm_in_kernel=True,use_gate_in_kernel=True,use_beta_sigmoid_in_kernel=True,state_v_first=True)
    with torch.no_grad():
     chunk,_=ORIGINAL[family,'chunk_gated_delta_rule'](**common)
     rec,_=ORIGINAL[family,'fused_recurrent_gated_delta_rule'](**common)
    alpha=(-record['A_log'].double().exp()[None,None]*F.softplus(raw.double().cpu()+record['dt_bias'].double()[None,None])).exp()[0].transpose(0,1)
    beta=bb.double().cpu().sigmoid()[0].transpose(0,1)
    yy,_,_=delta(F.normalize(a,dim=-1),F.normalize(b,dim=-1),c,alpha,beta)
    truth=yy.transpose(0,1)[None]
   chunk=chunk.double().cpu();rec=rec.double().cpu()
   # BF16 rounding is measured, never confused with FP64 algebra checks.
   err=float((chunk-truth).norm()/truth.norm());re=float((rec-truth).norm()/truth.norm())
   assert err<.035 and re<.035,(family,l,kind,err,re)
   rows.append(dict(layer=l,kind=kind,chunk_relative_l2=err,recurrent_relative_l2=re))
 save(P/'checks'/f'{family}_operators.json',dict(passed=True,rows=rows,
  scope='Actual native 128-token q/k/v/gates; BF16 FLA chunk and fused recurrence versus explicit CPU FP64 recurrence.'))
 print(family,'operator verification passed',flush=True)

if __name__=='__main__':main(sys.argv[1])
