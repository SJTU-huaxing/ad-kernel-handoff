import argparse
from core import *

def fit(data,norm,scale,val,kind,m,seed,n):
    name=f'{kind}_m{m}_s{seed}_n{n}';out=P/'fits'/f'{name}.json'
    if out.exists():print(json.dumps(dict(event='skip',name=name)),flush=True);return
    torch.manual_seed(seed);variant=kind if kind in ['factorized','factorized_both'] else 'exp' if kind=='exp_control' else 'softplus';model=Pair(norm,m,variant).cuda()
    order=torch.randperm(n,generator=torch.Generator().manual_seed(20260907+seed)).tolist()
    # Each document is visited once, with 64 distinct Q and all 512 distinct K.
    # All 32768 Q/K combinations occur once; no multi-epoch reuse.
    qgen=torch.Generator().manual_seed(20260908+seed)
    queries=torch.stack([torch.randperm(512,generator=qgen)[:64] for _ in range(n)]).cuda()
    optim=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,n,eta_min=.0002)
    start=time.perf_counter();history=[];running=torch.zeros(4,device='cuda')
    for step,i in enumerate(order,1):
        q=data['q'][i,:,queries[i]].float();k=data['k'][i].float()
        target=q@k.transpose(-1,-2)/math.sqrt(128)-scale[:,None,None]
        pred=model.log_matrix(q,k);perhead=loss_rows(pred,target,kind).mean(-1)
        if not torch.isfinite(perhead).all():raise RuntimeError((name,step,perhead))
        optim.zero_grad(set_to_none=True);perhead.mean().backward()
        sums=torch.zeros(4,device='cuda')
        for p in model.parameters():sums+=p.grad.flatten(1).square().sum(-1)
        factor=(10/sums.sqrt().clamp_min(1e-20)).clamp_max(1)
        for p in model.parameters():p.grad.mul_(factor.reshape(4,*([1]*(p.ndim-1))))
        optim.step();scheduler.step();running+=perhead.detach()
        if step%512==0 or step==n:
            row=dict(step=step,mean_loss=(running/(512 if step%512==0 else step%512)).tolist(),seconds=time.perf_counter()-start)
            history.append(row);running.zero_();print(json.dumps(dict(event='train',name=name,**row)),flush=True)
    seconds=time.perf_counter()-start
    meta=dict(name=name,kind=kind,variant=variant,m=m,seed=seed,training_documents=n,epochs=1,hidden_width=192,
        parameters_per_head=sum(p.numel() for p in model.parameters())//4,
        training_pairs_per_head=n*64*512,unique_query_vectors_per_head=n*64,
        document_order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),
        query_indices_sha256=hashlib.sha256(queries.cpu().numpy().tobytes()).hexdigest(),
        log_scale=scale.tolist(),lr=.002,weight_decay=1e-4,training_seconds=seconds,
        target=('Normalized attention weights, matched control for the Hedgehog-style distillation objective; raw scale unidentifiable.' if kind in ['kl_control','exp_control'] else 'exp(q^T k / sqrt(128)), with restored fixed per-head scale'),
        selection='Final single-pass checkpoint; no early stopping; validation only selects scenarios.')
    assert meta['parameters_per_head']==49152+386*m
    torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},metadata=meta),P/'fits'/f'{name}.pt')
    model.double().eval();metrics=evaluate(model,val,scale)
    save(out,dict(metadata=meta,history=history,validation=metrics))
    print(json.dumps(dict(event='complete',name=name,seconds=seconds,validation=metrics['summary'])),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kinds',nargs='+',default=['raw','balanced','half','logcosh'])
    ap.add_argument('--ranks',nargs='+',type=int,default=[64]);ap.add_argument('--seeds',nargs='+',type=int,default=[11])
    ap.add_argument('--n',type=int,default=4096);args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    manifest,data,norm,scale=load_train();val=load_full('validation')
    print(json.dumps(dict(event='loaded',documents=len(data['q']),scale=scale.tolist())),flush=True)
    for kind in args.kinds:
        for m in args.ranks:
            for seed in args.seeds:fit(data,norm,scale,val,kind,m,seed,args.n)

if __name__=='__main__':main()
