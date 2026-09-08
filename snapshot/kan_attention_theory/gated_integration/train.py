import argparse, gc, traceback
import native as n
from native import *

SEEDS=[11,29]

def base_fingerprint(model):
 h=hashlib.sha256()
 for name,p in model.named_parameters():
  h.update(name.encode());h.update(p.detach().cpu().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()

def run(family,followup=False):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 model=model_load(family)
 data=data_load()['records'];train=[r for r in data if r['split']=='train'];val=[r for r in data if r['split']=='validation']
 norms=collect_norm(model,family,train)
 fingerprint=base_fingerprint(model)
 kinds=['signed_residual'] if followup else KINDS[family]
 suffix='_followup' if followup else ''
 save(P/'results'/f'{family}{suffix}_plan.json',dict(family=family,kinds=kinds,seeds=SEEDS,
  training_docs=len(train),batch_size=4,steps=len(train)//4,layers=LAYERS,
  parameter_count_per_head=2*(norms[str(LAYERS[0])]['q_mean'].shape[-1]*192+192*norms[str(LAYERS[0])]['q_mean'].shape[-1]),
  base_sha256=fingerprint,objective='next-token cross entropy; frozen native model; not exp kernel distillation'))
 for kind in kinds:
  for seed in SEEDS:
   name=f'{family}_{kind}_s{seed}';path=P/'fits'/f'{name}.pt'
   if path.exists():continue
   maps=adapters(model,family,kind,norms,seed)
   params=list(maps.parameters());count=sum(p.numel() for p in params)
   assert not any(p.requires_grad for p in model.parameters())
   optim=torch.optim.AdamW(params,lr=.001,weight_decay=.0001)
   steps=len(train)//4;sched=torch.optim.lr_scheduler.CosineAnnealingLR(optim,steps,eta_min=.0001)
   order=torch.randperm(len(train),generator=torch.Generator().manual_seed(seed+98000)).tolist()
   initial=evaluate(model,val);history=[];running=0;start=time.perf_counter();grad_first=None
   print(json.dumps(dict(event='initial',name=name,params=count,validation=initial['ppl'])),flush=True)
   for step in range(steps):
    part=[train[i] for i in order[step*4:(step+1)*4]]
    x=torch.tensor([r['input_ids'] for r in part],device='cuda')
    optim.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
     logits=model(x,use_cache=False).logits
     loss=F.cross_entropy(logits[:,:-1].float().flatten(0,1),x[:,1:].flatten())
    assert torch.isfinite(loss),(name,step,float(loss))
    loss.backward()
    norm=nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    if step==0:
     grad_first={key:float(p.grad.float().norm()) if p.grad is not None else None for key,p in maps.named_parameters()}
    optim.step();sched.step();running+=float(loss.detach())
    del logits,loss
    if (step+1)%64==0:
     vv=evaluate(model,val)
     row=dict(step=step+1,train_nll=running/64,validation_ppl=vv['ppl'],validation_nll=vv['nll_per_token'],
      elapsed_seconds=time.perf_counter()-start,last_gradient_norm=float(norm))
     history.append(row);running=0
     print(json.dumps(dict(event='train',name=name,**row)),flush=True)
   metadata=dict(name=name,family=family,kind=kind,seed=seed,params=count,training_docs=len(train),steps=steps,
    training_tokens=len(train)*512,prediction_targets=len(train)*511,epochs=1,
    order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),
    initial_validation=initial,history=history,first_gradient_norms=grad_first,elapsed_seconds=time.perf_counter()-start)
   torch.save(dict(state_dict=maps.state_dict(),metadata=metadata),path)
   save(P/'fits'/f'{name}.json',metadata)
   n.MAPS=None;del maps,params,optim,sched;gc.collect();torch.cuda.empty_cache()
 assert base_fingerprint(model)==fingerprint
 save(P/'checks'/f'{family}_frozen.json',dict(passed=True,base_sha256=fingerprint,all_base_requires_grad_false=True))
 print(json.dumps(dict(event='finished',family=family)),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('family',choices=['gla','gdn']);ap.add_argument('--followup',action='store_true');a=ap.parse_args()
 try:run(a.family,a.followup)
 except Exception:
  (P/'logs'/f'{a.family}_training_error.txt').write_text(traceback.format_exc());raise
