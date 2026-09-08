"""Disjoint-query remote raw-mass calibration plus matched mixture-KL control."""
from common import *

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 ds=data('train','cuda');cal=data('calibration','cuda');index=KI.cuda();keys=torch.arange(1024,device='cuda')
 for seed in [11,29,47]:
  if (P/'fits'/f'gate_s{seed}.json').exists():continue
  base,bmeta=load(f'exp_kl_s{seed}',torch.float32);scale=torch.tensor(bmeta['log_scale'],device='cuda')
  with torch.no_grad():
   # Fixed train-only key feature means; feature rescaling is a paired gauge.
   bank=ds['k'][:,:,::256].permute(1,0,2,3).flatten(1,2).float()[index]
   mu=base.log_feature(bank,'k').double().logsumexp(1)-math.log(bank.shape[1]);del bank
   hs=[];offsets=[];oldlogs=[];targetlogs=[];locallogs=[];start=time.perf_counter();su=torch.zeros(H,192,device='cuda',dtype=torch.float64);sq=su.clone()
   for i in range(len(cal['q'])):
    q=cal['q'][i].float();k=ds['k'][i].float()[index];pos=cal['query_positions'][i];remote=(keys[None]<=pos[:,None]-64)[None];local=((keys[None]>pos[:,None]-64)&(keys[None]<=pos[:,None]))[None]
    hidden=F.silu((q-base.q_mean[:,None])/base.q_std[:,None]@base.qnet.w1.transpose(-1,-2))
    lt=q@k.transpose(-1,-2)/math.sqrt(D)-scale[:,None,None];lp=base.log_matrix(q,k)
    tm=lt.masked_fill(~remote,-torch.inf).logsumexp(-1);gm=lp.masked_fill(~remote,-torch.inf).logsumexp(-1);lm=lt.masked_fill(~local,-torch.inf).logsumexp(-1)
    ref=(base.log_feature(q,'q').double()+mu[:,None]).logsumexp(-1)
    hs.append(hidden.cpu());offsets.append((gm-ref-tm).cpu());oldlogs.append(gm.cpu());targetlogs.append(tm.cpu());locallogs.append(lm.cpu());su+=hidden.double().sum(1);sq+=hidden.double().square().sum(1)
   h=torch.stack(hs,1).cuda();del hs
   off=torch.stack(offsets,1).cuda().float();tm=torch.stack(targetlogs,1).double();gm=torch.stack(oldlogs,1).double();lm=torch.stack(locallogs,1).cuda().float();del offsets,oldlogs,targetlogs,locallogs
   hm=(su/(4096*64)).float();sd=(sq/(4096*64)-hm.double().square()).clamp_min(.03**2).sqrt().float();h.sub_(hm[:,None,None]).div_(sd[:,None,None])
   initial=-(off.double().flatten(1).logsumexp(-1)-math.log(4096*64)).float()
   local_truth=(lm-tm.cuda()).float();alpha=local_truth.sigmoid()
  for kind in ['calibrated','gate']:
   name=f'{kind}_s{seed}';u=nn.Parameter(torch.zeros(H,192,device='cuda'));bias=nn.Parameter(initial.clone());opt=torch.optim.Adam([u,bias],lr=.01);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,4096,eta_min=.001)
   order=torch.randperm(4096,generator=torch.Generator().manual_seed(86000+seed)).tolist();history=[]
   for step,i in enumerate(order,1):
    residual=off[:,i]+(h[:,i]*u[:,None]).sum(-1)+bias[:,None]
    if kind=='calibrated':loss=(torch.expm1(residual)-residual).mean(-1)
    else:
     predicted_local_logodds=local_truth[:,i]-residual
     loss=F.binary_cross_entropy_with_logits(predicted_local_logodds,alpha[:,i],reduction='none').mean(-1)
    loss=loss+1e-4*u.square().mean(-1);assert torch.isfinite(loss).all()
    opt.zero_grad(set_to_none=True);loss.mean().backward();fac=(10/(u.grad.square().sum(-1)+bias.grad.square()).sqrt().clamp_min(1e-20)).clamp_max(1);u.grad.mul_(fac[:,None]);bias.grad.mul_(fac);opt.step();sched.step()
    if step%1024==0:history.append(dict(step=step,loss=loss.tolist()))
   with torch.no_grad():
    w=u.double()/sd.double();b=bias.double()-(w*hm).sum(-1);state={k:v.detach().cpu().double().clone() for k,v in base.state_dict().items()};oldw=state['qnet.w2'];oldb=state['qnet.bias']+mu.cpu()
    state['qnet.w2']=torch.cat([oldw[:,:-1]-oldw[:,-1:],w.cpu()[:,None]],1)
    state['qnet.bias']=torch.cat([oldb[:,:-1]-oldb[:,-1:],(b.cpu()-.5*math.log(64))[:,None]],1);state['knet.bias']-=mu.cpu()
    meta=dict(bmeta,name=name,kind=kind,variant='gauge_calibrated',calibration_readout_parameters_per_head=193,calibration_queries_per_head=4096*64,calibration_pairs_per_head=int((cal['query_positions']-63).sum()),calibration_documents_revisited=True,calibration_pairs_disjoint=True,calibration_steps=4096,parameters_per_head=73856,calibration_seconds=time.perf_counter()-start,calibration_objective='raw remote mass I-divergence' if kind=='calibrated' else 'teacher exact/remote Bernoulli mixture KL',log_key_feature_means=mu.tolist(),calibration_note='Kernel targets evaluated once and cached; hidden mean/std and initialization computed from this calibration set before a single SGD pass. Same calibration data, architecture, initial readout, optimizer and update count for raw and mixture-KL controls; extra computation beyond one-stage training.')
    torch.save(dict(state_dict=state,metadata=meta),P/'fits'/f'{name}.pt');save(P/'fits'/f'{name}.json',dict(metadata=meta,history=history));print(json.dumps(dict(event='calibrated',name=name,seconds=time.perf_counter()-start)),flush=True)
  for kind,shift in dict(global_bal=-(gm-tm).flatten(1).logsumexp(-1)+math.log(4096*64),global_raw=tm.flatten(1).logsumexp(-1)-gm.flatten(1).logsumexp(-1)).items():
   name=f'{kind}_s{seed}';state={k:v.detach().cpu().double().clone() for k,v in base.state_dict().items()};state['qnet.bias']+=shift[:,None];meta=dict(bmeta,name=name,kind=kind,variant='exp',calibration_log_shift=shift.tolist(),calibration_steps=0,calibration_objective=kind,calibration_queries_per_head=4096*64,calibration_documents_revisited=True,calibration_pairs_disjoint=True)
   torch.save(dict(state_dict=state,metadata=meta),P/'fits'/f'{name}.pt');save(P/'fits'/f'{name}.json',dict(metadata=meta))
  del base,h,off,tm,gm,lm,local_truth,alpha
 # Fixed FAVOR+ orthogonal Gaussian feature maps, with raw target scaling.
 for seed in [11,29,47]:
  base,bmeta=load(f'exp_kl_s{seed}',torch.float32);norm={k:base.state_dict()[k] for k in ['q_mean','q_std','k_mean','k_std']};torch.manual_seed(90000+seed);net=Baseline(norm,'learned_prf').cuda();net.bias.data-=torch.tensor(bmeta['log_scale'],device='cuda')[:,None]
  name=f'favor_s{seed}';meta=dict(bmeta,name=name,kind='favor',variant='learned_prf',parameters_per_head=0,stored_random_scalars_per_head=8256,training_steps=0,training_seconds=0.,scope='Orthogonal positive Gaussian FAVOR+ feature formula, fixed m64; no training, same state budget.')
  torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},metadata=meta),P/'fits'/f'{name}.pt');save(P/'fits'/f'{name}.json',dict(metadata=meta))
 plan=json.loads((P/'results/frozen_plan.json').read_text());plan.update(calibration_kinds=['calibrated','gate','global_bal','global_raw'],window=64,calibration_frozen=True);save(P/'results/frozen_plan.json',plan)

if __name__=='__main__':main()
