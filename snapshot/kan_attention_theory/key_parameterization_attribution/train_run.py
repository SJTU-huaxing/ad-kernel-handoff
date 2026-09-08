import time
from core_attribution import *

def train(init,kind,seed,lr,ds,val,norm,scale):
    name=new_name(init,kind,seed,lr);path=P/'fits'/f'{name}.json'
    if path.exists():return read(path)
    torch.manual_seed(seed);net=Features(norm,kind,init,scale).cuda()
    q_hash=fingerprint(net.qnet.state_dict());k_hash=fingerprint(net.knet.state_dict())
    n=len(ds['q']);optim=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(optim,n,eta_min=lr/10)
    order=torch.randperm(n,generator=torch.Generator().manual_seed(85000+seed)).tolist()
    index=KI.cuda();keys=torch.arange(1024,device='cuda')
    running=torch.zeros(H,device='cuda');history=[];start=time.perf_counter()
    for step,i in enumerate(order,1):
        q=ds['q'][i].float();k=ds['k'][i].float()[index]
        lt=q@k.transpose(-1,-2)/math.sqrt(D)-scale[:,None,None]
        lp=net.log_matrix(q,k)
        mask=(keys[None]<=ds['query_positions'][i,:,None])[None]
        loss,_,_=source.rows_loss(lp,lt,mask,0.)
        perhead=loss.mean(-1)
        assert torch.isfinite(perhead).all(),(name,step)
        optim.zero_grad(set_to_none=True);perhead.mean().backward()
        sums=torch.zeros(H,device='cuda')
        for par in net.parameters():
            assert par.grad is not None
            sums+=par.grad.flatten(1).square().sum(-1)
        factor=(10/sums.sqrt().clamp_min(1e-20)).clamp_max(1)
        for par in net.parameters():par.grad.mul_(factor.reshape(H,*([1]*(par.ndim-1))))
        optim.step();sched.step();running+=perhead.detach()
        if step%512==0:
            row=dict(step=step,per_head_loss=(running/512).tolist(),seconds=time.perf_counter()-start)
            history.append(row);running.zero_()
            print(json.dumps(dict(event='train',name=name,step=step,loss=sum(row['per_head_loss'])/H,seconds=row['seconds'])),flush=True)
    torch.cuda.synchronize()
    meta=dict(name=name,map_kind=kind,initialization=init,seed=seed,lr=lr,m=64,parameters_per_head=73536,
        heads=HEADS,train_documents=n,training_queries_per_head=n*64,
        training_pairs_per_head=int((ds['query_positions']+1).sum()),epochs=1,
        log_scale=scale.tolist(),order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),
        query_sha256=hashlib.sha256(ds['query_positions'].cpu().numpy().tobytes()).hexdigest(),
        q_initial_sha256=q_hash,k_initial_sha256=k_hash,training_seconds=time.perf_counter()-start,
        objective='Causal teacher attention KL only; no amplitude or output auxiliary loss',
        protocol_sha256=hashlib.sha256((P/'PROTOCOL.zh.md').read_bytes()).hexdigest())
    torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},metadata=meta),P/'fits'/f'{name}.pt')
    metric=parent.evaluate(net.double().eval(),val,scale)
    score=sum(metric['summary']['kl'])/H
    out=dict(metadata=meta,history=history,validation=metric,selection_score=score)
    save(path,out);print(json.dumps(dict(event='fit',name=name,validation_kl=score)),flush=True)
    return out

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    ds=source.data('train','cuda');val={k:v[:32] for k,v in source.data('validation').items()}
    norm,scale=source.norm_scale(ds)
    selected={};names=[]
    for init,kind in CONFIGS:
        trials=[train(init,kind,11,lr,ds,val,norm,scale) for lr in [.002,.0005]]
        lr=min(trials,key=lambda v:v['selection_score'])['metadata']['lr'];selected[f'{init}_{kind}']=lr
        save(P/'results/selection.json',dict(learning_rates=selected))
        for seed in [29,47]:train(init,kind,seed,lr,ds,val,norm,scale)
        names.extend(new_name(init,kind,s,lr) for s in SEEDS)
    save(P/'results/plan.json',dict(models=names,learning_rates=selected,seeds=SEEDS,
        scope='9new selected models;3old standardAD references; fixed budget KL attribution'))

if __name__=='__main__':main()
