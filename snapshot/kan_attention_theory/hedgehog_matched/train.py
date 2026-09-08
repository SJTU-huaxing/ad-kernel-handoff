import hashlib
import time

from models import *


def run(kind, seed, lr, ds, val, norm, scale):
    name = f'{kind}_s{seed}_lr{lr:g}'
    out = P / 'fits' / (name + '.json')
    if out.exists():
        return json.loads(out.read_text())
    torch.manual_seed(seed)
    net = Matched(norm, kind).cuda()
    params = sum(p.numel() for p in net.parameters()) // H
    assert params == (73729 if kind.endswith('_raw') else 73728)
    lam = 1. if kind.endswith('_raw') else 0.
    optim = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    n = len(ds['q'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n, eta_min=lr / 10)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(85000 + seed)).tolist()
    keys = torch.arange(1024, device='cuda')
    index = KI.cuda()
    running = torch.zeros(H, device='cuda')
    history = []
    start = time.perf_counter()
    for step, i in enumerate(order, 1):
        q, k = ds['q'][i].float(), ds['k'][i].float()[index]
        mask = (keys[None] <= ds['query_positions'][i, :, None])[None]
        lt = q @ k.transpose(-1, -2) / math.sqrt(D) - scale[:, None, None]
        lp = net.log_matrix(q, k)
        loss, _, _ = source.rows_loss(lp, lt, mask, lam)
        perhead = loss.mean(-1)
        assert torch.isfinite(perhead).all(), (name, step)
        optim.zero_grad(set_to_none=True)
        perhead.mean().backward()
        sums = torch.zeros(H, device='cuda')
        for p in net.parameters():
            sums += p.grad.flatten(1).square().sum(-1) if p.ndim > 1 else p.grad.square()
        factor = (10 / sums.sqrt().clamp_min(1e-20)).clamp_max(1)
        for p in net.parameters():
            p.grad.mul_(factor.reshape(H, *([1] * (p.ndim - 1))))
        optim.step()
        sched.step()
        running += perhead.detach()
        if step % 512 == 0:
            row = dict(step=step, per_head_loss=(running / 512).tolist(),
                       seconds=time.perf_counter() - start)
            history.append(row)
            print(json.dumps(dict(event='train', name=name, step=step,
                                  loss=float((running / 512).mean()), seconds=row['seconds'])), flush=True)
            running.zero_()
    torch.cuda.synchronize()
    meta = dict(name=name, kind=kind, seed=seed, lr=lr, m=net.m,
                parameters_per_head=params, heads=HEADS, lambda_mass=lam,
                training_seconds=time.perf_counter() - start, train_documents=n,
                training_pairs_per_head=int((ds['query_positions'] + 1).sum()),
                training_queries_per_head=n * 64, epochs=1,
                log_scale=scale.tolist(),
                order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),
                query_sha256=hashlib.sha256(ds['query_positions'].cpu().numpy().tobytes()).hexdigest())
    torch.save(dict(state_dict={key: value.cpu() for key, value in net.state_dict().items()},
                    metadata=meta), P / 'fits' / (name + '.pt'))
    metrics = evaluate(net.double().eval(), val, scale)
    objective = 'balanced' if lam else 'kl'
    record = dict(metadata=meta, history=history, validation=metrics,
                  selection_score=sum(metrics['summary'][objective]) / H)
    save(out, record)
    print(json.dumps(dict(event='fit', name=name, selection_score=record['selection_score'])), flush=True)
    return record


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    ds = source.data('train', 'cuda')
    val = {key: value[:32] for key, value in source.data('validation').items()}
    norm, scale = source.norm_scale(ds)
    print(json.dumps(dict(event='loaded', q=list(ds['q'].shape), k=list(ds['k'].shape))), flush=True)
    selected = {}
    for kind in KINDS:
        trials = [run(kind, 11, lr, ds, val, norm, scale) for lr in [.002, .0005]]
        winner = min(trials, key=lambda x: x['selection_score'])
        selected[kind] = winner['metadata']['lr']
    save(P / 'results' / 'selection.json',
         dict(learning_rates=selected, criterion='Training objective on fixed first32 validation docs.',
              protocol_sha256=hashlib.sha256((P / 'PROTOCOL.zh.md').read_bytes()).hexdigest()))
    for kind in KINDS:
        for seed in [29, 47]:
            run(kind, seed, selected[kind], ds, val, norm, scale)
    save(P / 'results' / 'frozen_plan.json', dict(
        models=[f'{kind}_s{seed}_lr{selected[kind]:g}' for kind in KINDS for seed in SEEDS],
        kinds=KINDS, seeds=SEEDS, learning_rates=selected, parameter_matching='Exact within each objective',
        scope='Frozen Qwen2.5-1.5B,24 heads; no pretrained model weights trained.'))


if __name__ == '__main__':
    main()
