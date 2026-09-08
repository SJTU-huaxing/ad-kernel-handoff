import argparse
import copy
import gc
import time
import os

from models import *


def plan():
    return json.loads((P / 'results' / os.environ.get('MATCHED_PLAN', 'frozen_plan.json')).read_text())


def check_path(stem):
    suffix = '_product' if os.environ.get('MATCHED_PLAN') == 'product_plan.json' else ''
    return P / 'checks' / (stem + suffix + '.json')


@torch.inference_mode()
def kernels():
    for split in ['confirm_wiki', 'confirm_long']:
        ds = source.data(split)
        for name in plan()['models']:
            path = P / 'results' / f'kernel_{split}_{name}.json'
            if path.exists():
                continue
            net, meta = load_fit(name)
            out = evaluate(net, ds, torch.tensor(meta['log_scale']))
            save(path, dict(name=name, split=split, metadata=meta, **out))
            print(json.dumps(dict(event='kernel', split=split, name=name,
                                  **{key: sum(out['summary'][key]) / H
                                     for key in ['balanced', 'kl', 'output_nmse', 'raw_nmse']})), flush=True)
            del net


@torch.inference_mode()
def products():
    # All selected Q/K pairs; documents on the two sides are disjoint.
    ds = source.data('confirm_wiki')
    for direction, qa, ka in [('ab', slice(0, 64), slice(64, 128)),
                              ('ba', slice(64, 128), slice(0, 64))]:
        q = ds['q'][qa, :, ::4].permute(1, 0, 2, 3).flatten(1, 2).cuda().double()
        k = ds['k'][ka, :, ::16].permute(1, 0, 2, 3).flatten(1, 2).cuda().double()[KI.cuda()]
        for name in plan()['models']:
            path = P / 'results' / f'product_{direction}_{name}.json'
            if path.exists():
                continue
            net, meta = load_fit(name)
            scale = torch.tensor(meta['log_scale'], device='cuda', dtype=torch.float64)
            lq = net.log_feature(q, 'q')
            lk = net.log_feature(k, 'k')
            sums = {key: torch.zeros(H, device='cuda', dtype=torch.float64)
                    for key in ['raw_sse', 'raw_energy', 'raw_i', 'raw_truth_mass',
                                'kl', 'mass', 'balanced']}
            per_doc = {key: [] for key in ['raw_i', 'raw_truth_mass', 'raw_sse', 'raw_energy']}
            for start in range(0, q.shape[1], 128):
                logt = q[:, start:start+128] @ k.transpose(-1, -2) / math.sqrt(D) - scale[:, None, None]
                a, b = lq[:, start:start+128], lk
                aq, bk = a.amax(-1, keepdim=True), b.amax(-1, keepdim=True)
                logp = ((a-aq).exp() @ (b-bk).exp().transpose(-1, -2)).clamp_min(1e-300).log() + aq + bk.transpose(-1, -2)
                t, p = logt.exp(), logp.exp()
                mask = torch.ones_like(logt, dtype=torch.bool)
                balanced, kl, mass = source.rows_loss(logp, logt, mask, 1.)
                values = dict(raw_sse=(t-p).square().sum((-1, -2)),
                              raw_energy=t.square().sum((-1, -2)),
                              raw_i=(p-t+t*(logt-logp)).sum((-1, -2)),
                              raw_truth_mass=t.sum((-1, -2)),
                              kl=kl.sum(-1), mass=mass.sum(-1), balanced=balanced.sum(-1))
                cells = dict(raw_i=p-t+t*(logt-logp), raw_truth_mass=t,
                             raw_sse=(t-p).square(), raw_energy=t.square())
                for key, cell in cells.items():
                    per_doc[key].append(cell.reshape(H, -1, 16, k.shape[1]).sum((-1, -2)).cpu())
                for key, value in values.items():
                    sums[key] += value
            summary = {key: value.tolist() for key, value in sums.items()}
            for a, b, key in [('raw_sse', 'raw_energy', 'raw_nmse'),
                              ('raw_i', 'raw_truth_mass', 'relative_raw_i')]:
                summary[key] = (sums[a] / sums[b]).tolist()
            for key in ['kl', 'mass', 'balanced']:
                summary[key] = (sums[key] / q.shape[1]).tolist()
            assert all(math.isfinite(v) for vv in summary.values() for v in vv)
            save(path, dict(name=name, direction=direction, summary=summary,
                            per_query_document={key: torch.cat(value, 1).tolist()
                                                for key, value in per_doc.items()},
                            q_vectors=q.shape[1], k_vectors=k.shape[1],
                            q_docs=64, k_docs=64,
                            scope='Fixed empirical marginal product with disjoint documents; '
                                  'all selected pairs, no test fitting; not all cached positions '
                                  'or an unknown-population bound.'))
            print(json.dumps(dict(event='product', name=name, direction=direction,
                                  raw_i=sum(summary['relative_raw_i'])/H,
                                  raw_nmse=sum(summary['raw_nmse'])/H)), flush=True)
            del net


class Replacement:
    def __init__(self, original, name, dtype=torch.float32):
        self.original = original
        self.name = name
        self.maps = {}
        self.states = {}
        self.zero_den = 0
        if name == 'teacher':
            return
        full, self.meta = load_fit(name, dtype)
        for layer, start in [(14, 0), (27, 12)]:
            net = copy.deepcopy(full)
            for module in net.modules():
                for key, value in list(module._parameters.items()):
                    if value is not None:
                        module._parameters[key] = nn.Parameter(value[start:start+12].contiguous(),
                                                                requires_grad=False)
                for key, value in list(module._buffers.items()):
                    if value is not None and value.ndim and value.shape[0] == 24:
                        module._buffers[key] = value[start:start+12].contiguous()
            net.heads = 12
            self.maps[layer] = net
        del full

    def reset(self):
        self.states = {}
        self.zero_den = 0

    @torch.inference_mode()
    def __call__(self, module, q, k, v, mask, **kw):
        from operators import prefill, step
        layer = module.layer_idx
        if layer not in self.maps:
            return self.original(module, q, k, v, mask, **kw)
        assert q.shape[0] == 1 and k.shape[2] == q.shape[2]
        net = self.maps[layer]
        dtype = next(net.parameters()).dtype
        lq = net.log_feature(q[0].to(dtype), 'q')
        lk = net.log_feature(k[0].to(dtype).repeat_interleave(6, 0), 'k')
        if q.shape[2] == 1 and layer in self.states:
            y, den = step(lq, lk, v[0].to(dtype), self.states[layer], optimized=False)
        else:
            y, den, self.states[layer] = prefill(
                lq, lk, v[0].to(dtype).repeat_interleave(6, 0), self.states.get(layer))
        assert torch.isfinite(y).all() and (den > 0).all(), (self.name, layer)
        return y[None].transpose(1, 2).to(q.dtype).contiguous(), None


@torch.inference_mode()
def verify():
    from operators import prefill, step
    ds = source.data('confirm_wiki')
    rows = []
    for name in plan()['models']:
        net, meta = load_fit(name)
        # Actual 128 consecutive keys used as both test q/k coordinates;
        # Q vectors separately come from recorded Q, ensuring correct dimensions.
        q = ds['q'][0, :, :64].cuda().double()
        k = ds['k'][0, :, :64].cuda().double()[KI.cuda()]
        v = ds['v'][0, :, :64].cuda().double()[KI.cuda()]
        lq, lk = net.log_feature(q, 'q'), net.log_feature(k, 'k')
        lp = net.log_matrix(q, k)
        mask = torch.arange(64, device='cuda')[None, :] <= torch.arange(64, device='cuda')[:, None]
        truth = lp.masked_fill(~mask[None], -torch.inf).softmax(-1) @ v
        scan, den, state = prefill(lq, lk, v, chunk=16)
        error = float((scan-truth).abs().max() / truth.abs().max())
        assert error < 1e-10 and (den > 0).all()
        state2 = None
        ys = []
        for i in range(64):
            yi, _, state2 = prefill(lq[:, i:i+1], lk[:, i:i+1], v[:, i:i+1], state2, chunk=1)
            ys.append(yi)
        recurrent_error = float((torch.cat(ys, 1)-truth).abs().max() / truth.abs().max())
        assert recurrent_error < 1e-10
        rows.append(dict(name=name, params_per_head=sum(p.numel() for p in net.parameters())//H,
                         m=net.m, fp64_scan_relative_error=error,
                         fp64_recurrence_relative_error=recurrent_error))
    save(check_path('operators'), dict(passed=True, rows=rows))


@torch.inference_mode()
def precision():
    from operators import prefill
    ds = source.data('confirm_long')
    rows = []
    for name in [n for n in plan()['models'] if '_s11_' in n]:
        net = Replacement(None, name, torch.float64).maps[14]
        q = ds['q'][0, :12].cuda().double()
        k = ds['k'][0, :2].cuda().double().repeat_interleave(6, 0)
        v = ds['v'][0, :2].cuda().double().repeat_interleave(6, 0)
        positions = ds['query_positions'][0].cuda()
        mask = torch.arange(k.shape[1], device='cuda')[None] <= positions[:, None]
        expected = net.log_matrix(q, k).masked_fill(~mask[None], -torch.inf).softmax(-1) @ v
        net.float()
        lq = net.log_feature(q.float(), 'q')
        lk = net.log_feature(k.float(), 'k')
        full_lq = torch.zeros(12, k.shape[1], net.m, device='cuda')
        full_lq[:, positions] = lq
        output, den, state = prefill(full_lq, lk, v.float())
        observed = output[:, positions].double()
        relative = float((observed-expected).abs().max()/expected.abs().max())
        assert relative < 1e-4 and (den > 0).all() and torch.isfinite(output).all(), (name, relative)
        rows.append(dict(name=name, n=k.shape[1], selected_queries=q.shape[1],
                         relative_max_output_error=relative,
                         zero_denominators=int((den<=0).sum()),
                         scope='FP32 full8k chunk scan against FP64 explicit real64Q x8192K,first12heads,firstheldoutlongdoc.'))
        del net, lq, lk, full_lq, output, state
        torch.cuda.empty_cache()
    save(check_path('long_precision'), dict(passed=True, rows=rows))


@torch.inference_mode()
def ppl():
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen2 import modeling_qwen2
    assert json.loads(check_path('operators').read_text())['passed']
    manifest = json.loads((ROOT/'causal_direction/data/manifest.json').read_text())
    original = modeling_qwen2.ALL_ATTENTION_FUNCTIONS['sdpa']
    model = AutoModelForCausalLM.from_pretrained(
        manifest['model'], revision=manifest['revision'],
        cache_dir='/root/autodl-tmp/hf-cache', local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    try:
        for name in ['teacher'] + plan()['models']:
            rt = Replacement(original, name)
            modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa', rt)
            for split in ['confirm_wiki', 'confirm_long']:
                path = P / 'results' / f'ppl_{split}_{name}.json'
                if path.exists():
                    continue
                records = [r for r in manifest['records'] if r['split'] == split]
                rows = []
                start = time.perf_counter()
                for i, record in enumerate(records):
                    ids = torch.tensor([record['input_ids']], device='cuda')
                    rt.reset()
                    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
                    loss = torch.zeros((), dtype=torch.float64, device='cuda')
                    for begin in range(0, ids.shape[1]-1, 256):
                        end = min(begin+256, ids.shape[1]-1)
                        logits = model.lm_head(hidden[:, begin:end]).float()
                        loss += F.cross_entropy(logits[0], ids[0, begin+1:end+1],
                                                reduction='none').double().sum()
                    rows.append(dict(ordinal=i, nll=float(loss), tokens=ids.shape[1]-1))
                tokens = sum(r['tokens'] for r in rows)
                nll = sum(r['nll'] for r in rows)
                out = dict(name=name, split=split, ppl=math.exp(nll/tokens),
                           nll_per_token=nll/tokens, documents=rows,
                           seconds=time.perf_counter()-start,
                           scope='Frozen full-model PPL;24/336 heads at two complete layers replaced; '
                                 'no model-weight finetuning; pure linear attention.')
                save(path, out)
                print(json.dumps({k:v for k,v in out.items() if k!='documents'}), flush=True)
            del rt
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa', original)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['kernels', 'products', 'verify', 'precision', 'ppl'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    globals()[args.action]()
