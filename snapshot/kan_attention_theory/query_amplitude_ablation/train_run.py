import hashlib
import time
from ablation import *


def run(regime, variant, seed, lr, ds, val, norm, scale):
    name = f'{regime}_{variant}_s{seed}_lr{lr:g}'
    path = P / 'fits' / f'{name}.json'
    if path.exists():
        return json.loads(path.read_text())
    torch.manual_seed(seed)
    net = Reduced(norm, variant, regime, scale).cuda()
    count = sum(p.numel() for p in net.parameters()) // H
    n = len(ds['q'])
    optim = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n, eta_min=lr/10)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(85000+seed)).tolist()
    index, keys = KI.cuda(), torch.arange(1024, device='cuda')
    history, running = [], torch.zeros(H, device='cuda')
    start = time.perf_counter()
    for step, i in enumerate(order, 1):
        q = ds['q'][i].float()
        k = product.keys_for(ds, i) if regime == 'product_i' else ds['k'][i].float()[index]
        lt = q @ k.transpose(-1, -2) / math.sqrt(D) - scale[:, None, None]
        lp = net.log_matrix(q, k)
        if regime == 'product_i':
            t, pred = lt.exp(), lp.exp()
            perhead = (pred-t+t*(lt-lp)).mean((-1, -2))
        else:
            mask = (keys[None] <= ds['query_positions'][i, :, None])[None]
            loss, _, _ = source.rows_loss(lp, lt, mask, 0. if regime == 'causal_kl' else 1.)
            perhead = loss.mean(-1)
        assert torch.isfinite(perhead).all(), (name, step)
        optim.zero_grad(set_to_none=True)
        perhead.mean().backward()
        sums = torch.zeros(H, device='cuda')
        for par in net.parameters():
            assert par.grad is not None
            sums += par.grad.flatten(1).square().sum(-1) if par.ndim > 1 else par.grad.square()
        factor = (10 / sums.sqrt().clamp_min(1e-20)).clamp_max(1)
        for par in net.parameters():
            par.grad.mul_(factor.reshape(H, *([1]*(par.ndim-1))))
        optim.step(); sched.step(); running += perhead.detach()
        if step % 512 == 0:
            row = dict(step=step, per_head_loss=(running/512).tolist(), seconds=time.perf_counter()-start)
            history.append(row)
            print(json.dumps(dict(event='train', name=name, step=step, loss=float((running/512).mean()), seconds=row['seconds'])), flush=True)
            running.zero_()
    torch.cuda.synchronize()
    meta = dict(name=name, regime=regime, variant=variant, kind=f'{regime}__{variant}',
                seed=seed, lr=lr, m=64, parameters_per_head=count, heads=HEADS,
                train_documents=n, training_queries_per_head=n*64,
                training_pairs_per_head=n*64*1024 if regime == 'product_i' else int((ds['query_positions']+1).sum()),
                epochs=1, log_scale=scale.tolist(),
                raw_function='exp(s_K(k))*simplex_Q(q).T@simplex_K(k); no query amplitude or outer C',
                order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),
                query_sha256=hashlib.sha256(ds['query_positions'].cpu().numpy().tobytes()).hexdigest(),
                training_seconds=time.perf_counter()-start)
    torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()}, metadata=meta), P/'fits'/f'{name}.pt')
    if regime == 'product_i':
        metric = product.validation(net, val, scale)
        score = metric['mean']
    else:
        metric = parent.evaluate(net.double().eval(), {k:v[:32] for k,v in val.items()}, scale)
        score = sum(metric['summary']['kl' if regime == 'causal_kl' else 'balanced']) / H
    out = dict(metadata=meta, history=history, validation=metric, selection_score=score)
    save(path, out)
    print(json.dumps(dict(event='fit', name=name, selection_score=score)), flush=True)
    return out


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    ds = source.data('train', 'cuda')
    val = {k:v[:64] for k,v in source.data('validation').items()}
    norm, causal_scale = source.norm_scale(ds)
    scales = {'product_i':product.scale_for(ds), 'causal_i':causal_scale, 'causal_kl':causal_scale}
    selected, names = {}, []
    for regime in REGIMES:
        for variant in VARIANTS:
            trials = [run(regime, variant, 11, lr, ds, val, norm, scales[regime]) for lr in [.002, .0005]]
            lr = min(trials, key=lambda x:x['selection_score'])['metadata']['lr']
            selected[f'{regime}__{variant}'] = lr
            save(P/'results'/'selection.json', dict(learning_rates=selected,
                 protocol_sha256=hashlib.sha256((P/'PROTOCOL.zh.md').read_bytes()).hexdigest()))
            for seed in [29,47]:
                run(regime, variant, seed, lr, ds, val, norm, scales[regime])
            names.extend(f'{regime}_{variant}_s{seed}_lr{lr:g}' for seed in SEEDS)
    save(P/'results'/'plan.json', dict(models=names, regimes=REGIMES, variants=VARIANTS,
         seeds=SEEDS, learning_rates=selected, new_fit_count=24,
         scope='Frozen Qwen2.5-1.5B;24 heads; one-pass function training, no LLM weight update.'))


if __name__ == '__main__':
    main()
