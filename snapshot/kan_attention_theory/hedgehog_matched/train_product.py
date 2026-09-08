import hashlib
import time
from models import *

OFFSETS = [127,251,509,761,1019,1279,1531,1789,2053,2309,2557,2819,3067,3323,3581,3833]


def keys_for(ds, i):
    ix = torch.tensor([(i+j) % len(ds['k']) for j in OFFSETS], device='cuda')
    # [16 docs,4 KVheads,64 positions,128] -> [24 Qheads,1024 keys,128]
    return ds['k'][ix, :, ::16].permute(1,0,2,3).flatten(1,2)[KI.cuda()].float()


@torch.inference_mode()
def scale_for(ds):
    path = P/'results/product_scale.json'
    if path.exists():
        return torch.tensor(json.loads(path.read_text())['log_scale'], device='cuda')
    total = torch.full((H,), -torch.inf, dtype=torch.float64, device='cuda')
    for i in range(len(ds['q'])):
        lt = ds['q'][i].double() @ keys_for(ds,i).double().transpose(-1,-2) / math.sqrt(D)
        total = torch.logaddexp(total, lt.flatten(1).logsumexp(-1))
    count = len(ds['q'])*64*1024
    scale = total-math.log(count)
    save(path, dict(log_scale=scale.tolist(), pairs_per_head=count, offsets=OFFSETS,
                    scope='Only fixed training product pairs; label/moment prepass, no parameter updates.'))
    return scale.float()


@torch.inference_mode()
def validation(net, val, scale):
    q = val['q'][:32,:,::2].permute(1,0,2,3).flatten(1,2).cuda().double()
    k = val['k'][32:64,:,::32].permute(1,0,2,3).flatten(1,2).cuda().double()[KI.cuda()]
    lt = q @ k.transpose(-1,-2)/math.sqrt(D)-scale.double()[:,None,None]
    lp = net.double().log_matrix(q,k)
    t,p = lt.exp(),lp.exp()
    div = (p-t+t*(lt-lp)).sum((-1,-2))/t.sum((-1,-2))
    assert torch.isfinite(div).all()
    return dict(relative_raw_i=div.tolist(), mean=float(div.mean()), q_vectors=q.shape[1], k_vectors=k.shape[1])


def run(kind,seed,lr,ds,val,norm,scale):
    name=f'product_{kind}_s{seed}_lr{lr:g}'
    path=P/'fits'/f'{name}.json'
    if path.exists():
        return json.loads(path.read_text())
    torch.manual_seed(seed)
    net=Matched(norm,kind).cuda()
    assert sum(p.numel() for p in net.parameters())//H == 73729
    n=len(ds['q'])
    optim=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(optim,n,eta_min=lr/10)
    order=torch.randperm(n,generator=torch.Generator().manual_seed(85000+seed)).tolist()
    history=[];running=torch.zeros(H,device='cuda');start=time.perf_counter()
    for step,i in enumerate(order,1):
        q,k=ds['q'][i].float(),keys_for(ds,i)
        lt=q@k.transpose(-1,-2)/math.sqrt(D)-scale[:,None,None]
        lp=net.log_matrix(q,k)
        t,p=lt.exp(),lp.exp()
        perhead=(p-t+t*(lt-lp)).mean((-1,-2))
        assert torch.isfinite(perhead).all(),(name,step)
        optim.zero_grad(set_to_none=True);perhead.mean().backward()
        sums=torch.zeros(H,device='cuda')
        for par in net.parameters():
            sums+=par.grad.flatten(1).square().sum(-1) if par.ndim>1 else par.grad.square()
        factor=(10/sums.sqrt().clamp_min(1e-20)).clamp_max(1)
        for par in net.parameters():par.grad.mul_(factor.reshape(H,*([1]*(par.ndim-1))))
        optim.step();sched.step();running+=perhead.detach()
        if step%512==0:
            row=dict(step=step,per_head_loss=(running/512).tolist(),seconds=time.perf_counter()-start)
            history.append(row)
            print(json.dumps(dict(event='train_product',name=name,step=step,
                                  loss=float((running/512).mean()),seconds=row['seconds'])),flush=True)
            running.zero_()
    torch.cuda.synchronize()
    meta=dict(name=name,kind=kind,seed=seed,lr=lr,m=net.m,parameters_per_head=73729,heads=HEADS,
              objective='Unweighted original I-divergence divided by fixed per-head training kernel mean.',
              train_documents=n,training_queries_per_head=n*64,training_pairs_per_head=n*64*1024,
              epochs=1,log_scale=scale.tolist(),offsets=OFFSETS,
              order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),
              query_sha256=hashlib.sha256(ds['query_positions'].cpu().numpy().tobytes()).hexdigest(),
              training_seconds=time.perf_counter()-start)
    torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},metadata=meta),
               P/'fits'/f'{name}.pt')
    metric=validation(net,val,scale)
    out=dict(metadata=meta,history=history,validation=metric,selection_score=metric['mean'])
    save(path,out);print(json.dumps(dict(event='fit_product',name=name,validation=metric['mean'])),flush=True)
    return out


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    ds=source.data('train','cuda');val={k:v[:64] for k,v in source.data('validation').items()}
    norm,_=source.norm_scale(ds);scale=scale_for(ds)
    rates={}
    for kind in ['ad_raw','hh_raw']:
        rows=[run(kind,11,lr,ds,val,norm,scale) for lr in [.002,.0005]]
        rates[kind]=min(rows,key=lambda r:r['selection_score'])['metadata']['lr']
    save(P/'results/product_selection.json',dict(
        learning_rates=rates,protocol_sha256=hashlib.sha256((P/'PRODUCT_PROTOCOL.zh.md').read_bytes()).hexdigest()))
    for kind in ['ad_raw','hh_raw']:
        for seed in [29,47]:run(kind,seed,rates[kind],ds,val,norm,scale)
    save(P/'results/product_plan.json',dict(
        models=[f'product_{kind}_s{seed}_lr{rates[kind]:g}' for kind in ['ad_raw','hh_raw'] for seed in SEEDS],
        kinds=['ad_raw','hh_raw'],seeds=SEEDS,learning_rates=rates,
        scope='Additional distribution-targeted matched-parameter experiment, disclosed after causal-stage product failure.'))


if __name__=='__main__':main()
