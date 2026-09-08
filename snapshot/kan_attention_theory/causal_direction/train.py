import argparse
from common import *
KINDS=['split_1','split_01','split_kl','exp_kl','softplus_kl','hedgehog','learned_prf']

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--kinds',nargs='+',default=KINDS);ap.add_argument('--seeds',nargs='+',type=int,default=[11,29,47]);a=ap.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 ds=data('train','cuda');val=data('validation');norm,scale=norm_scale(ds);print(json.dumps(dict(event='loaded',q=list(ds['q'].shape),k=list(ds['k'].shape),scale=scale.tolist())),flush=True)
 index=KI.cuda();keys=torch.arange(1024,device='cuda');n=len(ds['q'])
 for kind in a.kinds:
  for seed in a.seeds:
   name=f'{kind}_s{seed}';path=P/'fits'/f'{name}.json'
   if path.exists():continue
   torch.manual_seed(seed);net=make(norm,kind).cuda();lam=1. if kind=='split_1' else .1 if kind=='split_01' else 0.
   optim=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(optim,n,eta_min=.0002)
   order=torch.randperm(n,generator=torch.Generator().manual_seed(85000+seed)).tolist();start=time.perf_counter();hist=[];running=torch.zeros(H,device='cuda')
   for step,i in enumerate(order,1):
    q=ds['q'][i].float();k=ds['k'][i].float()[index];mask=(keys[None]<=ds['query_positions'][i,:,None])[None]
    lt=q@k.transpose(-1,-2)/math.sqrt(D)-scale[:,None,None];lp=net.log_matrix(q,k)
    loss,_,_=rows_loss(lp,lt,mask,lam);perhead=loss.mean(-1);assert torch.isfinite(perhead).all(),(kind,step)
    optim.zero_grad(set_to_none=True);perhead.mean().backward();sums=torch.zeros(H,device='cuda')
    for p in net.parameters():sums+=p.grad.flatten(1).square().sum(-1)
    fac=(10/sums.sqrt().clamp_min(1e-20)).clamp_max(1)
    for p in net.parameters():p.grad.mul_(fac.reshape(H,*([1]*(p.ndim-1))))
    optim.step();sched.step();running+=perhead.detach()
    if step%512==0:
     row=dict(step=step,loss=(running/512).tolist(),seconds=time.perf_counter()-start);hist.append(row);running.zero_();print(json.dumps(dict(event='train',name=name,**row)),flush=True)
   meta=dict(name=name,kind=kind,variant=net.variant,seed=seed,m=64,lambda_mass=lam,heads=HEADS,parameters_per_head=sum(p.numel() for p in net.parameters())//H,
    train_documents=n,epochs=1,training_queries_per_head=n*64,training_pairs_per_head=int((ds['query_positions']+1).sum()),
    order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),query_sha256=hashlib.sha256(ds['query_positions'].cpu().numpy().tobytes()).hexdigest(),
    log_scale=scale.tolist(),training_seconds=time.perf_counter()-start,lr=.002,weight_decay=1e-4,
    objective='KL(direction)+lambda*raw-mass I-divergence; lambda>0 retains raw target, lambda=0 explicitly normalized KL control.',
    scope='Full causal positions; final one-pass checkpoint; no frozen-LLM optimization. Published-form controls use their own parameter budget, not falsely parameter matched.')
   if kind not in ['hedgehog','learned_prf']:assert meta['parameters_per_head']==73856
   torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},metadata=meta),P/'fits'/f'{name}.pt')
   metrics=evaluate(net.double().eval(),val,scale);save(path,dict(metadata=meta,history=hist,validation=metrics));print(json.dumps(dict(event='fit',name=name,kl=sum(metrics['summary']['kl'])/H,mass=sum(metrics['summary']['mass'])/H,output=sum(metrics['summary']['output_nmse'])/H)),flush=True)
   del net,optim
 if all((P/'fits'/f'{k}_s{s}.json').exists() for k in KINDS for s in [11,29,47]):
  rows={k:[json.loads((P/'fits'/f'{k}_s{s}.json').read_text()) for s in [11,29,47]] for k in KINDS}
  selected=min(['split_1','split_01'],key=lambda k:sum(sum(r['validation']['summary']['output_nmse']) for r in rows[k]))
  save(P/'results/frozen_plan.json',dict(kinds=KINDS,seeds=[11,29,47],primary=selected,comparison='All configurations reported; primary lambda selected on validation only.',scope='24/336 heads at two complete layers.'))

if __name__=='__main__':main()
