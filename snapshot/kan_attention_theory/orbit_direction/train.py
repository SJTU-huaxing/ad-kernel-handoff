"""One bounded follow-up: same-budget common-RoPE augmentation, raw vs KL."""
import sys
from pathlib import Path
O=Path(__file__).resolve().parent;C=O.parent/'causal_direction';sys.path.insert(0,str(C))
from common import *

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;ds=data('train','cuda');val=data('validation');keys=torch.arange(1024,device='cuda');freq=1e6**(-torch.arange(0,128,2,device='cuda')/128)
 def rotate(x,phase):
  a,b=x[...,:64],x[...,64:];c=phase.cos();s=phase.sin();return torch.cat([a*c-b*s,b*c+a*s],-1)
 for seed in [11,29,47]:
  old=torch.load(C/'fits'/f'split_01_s{seed}.pt',weights_only=True);norm={k:old['state_dict'][k].cuda() for k in ['q_mean','q_std','k_mean','k_std']};scale=torch.tensor(old['metadata']['log_scale'],device='cuda');gen=torch.Generator().manual_seed(90100+seed);offsets=torch.randint(0,32769,(4096,),generator=gen);offsets[torch.rand(4096,generator=gen)<.5]=0;order=torch.randperm(4096,generator=torch.Generator().manual_seed(85000+seed)).tolist()
  for kind,lam in [('split_01',.1),('split_kl',0.)]:
   name=f'orbit_{kind}_s{seed}';path=O/'fits'/f'{name}.json'
   if path.exists():continue
   torch.manual_seed(seed);net=make(norm,kind).cuda();opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,4096,eta_min=.0002);running=torch.zeros(H,device='cuda');history=[];start=time.perf_counter()
   for step,i in enumerate(order,1):
    q0=ds['q'][i].float();k0=ds['k'][i].float();lt=q0@k0[KI.cuda()].transpose(-1,-2)/math.sqrt(D)-scale[:,None,None];phase=int(offsets[i])*freq;q=rotate(q0,phase);k=rotate(k0,phase)[KI.cuda()];mask=(keys[None]<=ds['query_positions'][i,:,None])[None];loss,_,_=rows_loss(net.log_matrix(q,k),lt,mask,lam);perhead=loss.mean(-1);assert torch.isfinite(perhead).all()
    opt.zero_grad(set_to_none=True);perhead.mean().backward();sums=torch.zeros(H,device='cuda')
    for p in net.parameters():sums+=p.grad.flatten(1).square().sum(-1)
    fac=(10/sums.sqrt().clamp_min(1e-20)).clamp_max(1)
    for p in net.parameters():p.grad.mul_(fac.reshape(H,*([1]*(p.ndim-1))))
    opt.step();sched.step();running+=perhead.detach()
    if step%512==0:history.append(dict(step=step,loss=(running/512).tolist()));running.zero_()
   meta=dict(old['metadata'],name=name,kind=kind,lambda_mass=lam,training_seconds=time.perf_counter()-start,augmentation='Same rotation for all Q and K in a document: delta0 with probability.5, otherwise integer uniform0..32768. Exactly preserves ideal raw target. Original unrotated target retained to avoid FP32 rotation rounding in labels.',offsets_sha256=hashlib.sha256(offsets.numpy().tobytes()).hexdigest(),offsets=offsets.tolist(),objective='Raw KL+0.1 mass I' if lam else 'Same architecture, pure directional KL control',same_original_normalization=True,same_optimizer_and_order=True,epochs=1)
   torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},metadata=meta),O/'fits'/f'{name}.pt');metrics=evaluate(net.double(),val,scale);save(path,dict(metadata=meta,history=history,validation=metrics));print(json.dumps(dict(event='fit',name=name,seconds=meta['training_seconds'],validation_kl=sum(metrics['summary']['kl'])/H,validation_output=sum(metrics['summary']['output_nmse'])/H)),flush=True);del net,opt
 save(O/'results/frozen_plan.json',dict(names=[f'orbit_{k}_s{s}' for k in ['split_01','split_kl'] for s in [11,29,47]],selection='No new hyperparameter selection: both predeclared objectives and all3seeds reported. Rotation protocol fixed from preceding causal_direction diagnostic; new confirmation texts never previously evaluated.',scope='Same24heads, m64,73856parameters/head,4096 updates, same base Q/K count, no LLM training.'))

if __name__=='__main__':main()
