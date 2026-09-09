"""Paired full-model latency benchmark for the existing AD/Hedgehog checkpoints.

Only the two trained layers are replaced. Both methods use the same FP32
reference scan/step and genuinely bypass their old KV caches. No retraining.
"""
import argparse
import copy
import gc
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch import nn
import transformers
from transformers import AutoModelForCausalLM
from transformers.models.qwen2 import modeling_qwen2

P = Path(__file__).resolve().parent
ROOT = P.parent
sys.path.insert(0, str(ROOT / 'query_amplitude_ablation'))
import ablation as ad
from operators import prefill, step
from runtime import cache_for, storage

CASES = {
    'teacher': None,
    'ad_plain': 'causal_kl_reduced_plain_s11_lr0.002',
    'ad_matched': 'causal_kl_reduced_matched_s11_lr0.002',
    'hh_exp': 'hh_kl_s11_lr0.002',
    'hh_softmax': 'hh_softmax_kl_s11_lr0.002',
}


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_snapshot():
    fields = 'temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw,utilization.gpu,memory.used'
    return subprocess.check_output(['nvidia-smi', '--query-gpu=' + fields,
                                    '--format=csv,noheader,nounits'], text=True).strip()


class Runtime:
    def __init__(self, original, case):
        self.original, self.case = original, case
        self.maps, self.states = {}, {}
        self.audit = False
        self.operator_checks = []
        self.metadata = {'case': case, 'checkpoint': CASES[case]}
        if case == 'teacher':
            return
        loader = ad.load_fit if case.startswith('ad_') else ad.parent.load_fit
        full, _ = loader(CASES[case], torch.float32)
        full.runtime_raw = True
        self.metadata.update(m=full.m, parameters_per_head=sum(p.numel() for p in full.parameters()) // 24)
        folder = ROOT / ('query_amplitude_ablation' if case.startswith('ad_') else 'hedgehog_matched')
        self.metadata['checkpoint_sha256'] = digest(folder / 'fits' / (CASES[case] + '.pt'))
        for layer, start in [(14, 0), (27, 12)]:
            net = copy.deepcopy(full)
            for module in net.modules():
                for key, value in list(module._parameters.items()):
                    if value is not None:
                        module._parameters[key] = nn.Parameter(value[start:start+12].clone().contiguous(),
                                                               requires_grad=False)
                for key, value in list(module._buffers.items()):
                    if value is not None and value.ndim and value.shape[0] == 24:
                        module._buffers[key] = value[start:start+12].clone().contiguous()
            net.heads = 12
            self.maps[layer] = net

    def reset(self):
        self.states = {}

    @torch.inference_mode()
    def __call__(self, module, q, k, v, mask, **kw):
        layer = module.layer_idx
        if layer not in self.maps:
            return self.original(module, q, k, v, mask, **kw)
        assert q.shape[0] == 1 and q.shape[2] == k.shape[2]
        net = self.maps[layer]
        lq = net.log_feature(q[0].float(), 'q')
        lk = net.log_feature(k[0].float().repeat_interleave(6, 0), 'k')
        if q.shape[2] == 1 and layer in self.states:
            y, den = step(lq, lk, v[0].float(), self.states[layer], optimized=False)
        else:
            y, den, self.states[layer] = prefill(
                lq, lk, v[0].float().repeat_interleave(6, 0), self.states.get(layer))
        if self.audit:
            assert torch.isfinite(y).all() and (den > 0).all()
            if q.shape[2] > 1:
                # Independent explicit kernel, using the same computed features.
                n = min(q.shape[2], 128)
                a, b = lq[:, :n].double(), lk[:, :n].double()
                aq, bk = a.amax(-1, keepdim=True), b.amax(-1, keepdim=True)
                lp = ((a-aq).exp() @ (b-bk).exp().transpose(-1, -2)).log()
                lp = lp + aq + bk.transpose(-1, -2)
                causal = torch.arange(n, device=q.device)[None] <= torch.arange(n, device=q.device)[:, None]
                expected = lp.masked_fill(~causal, -torch.inf).softmax(-1) @ v[0, :, :n].double().repeat_interleave(6, 0)
                rel = float((y[:, :n].double()-expected).norm()/expected.norm())
                assert rel < 2e-4, (self.case, layer, rel)
                self.operator_checks.append({'layer': layer, 'relative_l2': rel, 'tokens': n})
        return y[None].transpose(1, 2).to(q.dtype).contiguous(), None


def setup():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20260909)
    manifest_path = ROOT / 'causal_direction/data/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    original = modeling_qwen2.ALL_ATTENTION_FUNCTIONS['sdpa']
    model = AutoModelForCausalLM.from_pretrained(
        manifest['model'], revision=manifest['revision'],
        cache_dir='/root/autodl-tmp/hf-cache', local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    model.requires_grad_(False)
    runtimes = {name: Runtime(original, name) for name in CASES}
    docs = [r['input_ids'] for r in manifest['records'] if r['split'] == 'confirm_long']
    # Three real held-out documents; beyond 8k, continue with the next document.
    token_sets = [torch.tensor([docs[i] + docs[i+3][:256]], device='cuda') for i in range(3)]
    metadata = dict(
        utc_created=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        model=manifest['model'], revision=manifest['revision'],
        hardware=torch.cuda.get_device_name(), torch=str(torch.__version__),
        transformers=transformers.__version__, cuda=torch.version.cuda,
        python=platform.python_version(), cpu_threads=torch.get_num_threads(),
        base_model_parameters=sum(p.numel() for p in model.parameters()),
        batch_size=1, model_dtype='bfloat16', feature_state_dtype='float32',
        tf32=False, layers=[14, 27], replaced_heads=24, total_heads=336,
        cases={k: rt.metadata for k, rt in runtimes.items()},
        script_sha256=digest(Path(__file__)), manifest_sha256=digest(manifest_path),
        operator_sha256=digest(ROOT / 'causal_direction/operators.py'),
        cache_runtime_sha256=digest(ROOT / 'causal_direction/runtime.py'),
        token_sha256=[hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() for t in token_sets],
        protocol='HF eager model execution with SDPA for unchanged layers; identical FP32 reference '
                 'linear prefill/step for AD and Hedgehog; no per-token diagnostic synchronization. '
                 'Full model forward including LM head, logits_to_keep=1, actual DynamicCache '
                 'plus StateOnlyLayer at both replaced layers. Fixed-token continuation for fair '
                 'shapes/content, no text tokenization, sampling, request scheduling or network IO. '
                 'Not a full-head conversion, pretraining study, CUDA-graph serving engine, or NPU result.',
    )
    return model, runtimes, token_sets, original, metadata


def activate(rt):
    rt.reset()
    modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa', rt)


def forward(model, ids, cache=None):
    return model(input_ids=ids, past_key_values=cache, use_cache=cache is not None, logits_to_keep=1)


def validate_storage(cache, rt, n):
    result = storage(cache, rt)
    assert cache.get_seq_length() == n
    if rt.case != 'teacher':
        for layer in (14, 27):
            assert cache.layers[layer].keys.untyped_storage().nbytes() == 0
            assert cache.layers[layer].values.untyped_storage().nbytes() == 0
            assert cache.layers[layer].get_seq_length() == n
        expected = 2 * 12 * rt.metadata['m'] * (128 + 2) * 4
        assert result['state_bytes'] == expected, result
    return result


@torch.inference_mode()
def verify(model, runtimes, token_sets):
    rows = []
    def check(case, rt, total, dtype_name):
        activate(rt)
        rt.audit = total < 1024
        ids = token_sets[0][:, :total]
        whole = forward(model, ids).logits.float().clone()
        rt.audit = False
        rt.reset()
        cache = cache_for(model, case != 'teacher')
        prefix = total-16
        out = forward(model, ids[:, :prefix], cache)
        before = validate_storage(cache, rt, prefix)
        for t in range(prefix, total):
            out = forward(model, ids[:, t:t+1], cache)
        rel = float((whole-out.logits.float()).norm()/whole.norm())
        if dtype_name == 'float32':
            assert rel < 1e-3, (case, total, rel)
        after = validate_storage(cache, rt, total)
        assert before['state_bytes'] == after['state_bytes']
        assert torch.isfinite(out.logits).all()
        rows.append(dict(case=case, model_dtype=dtype_name, total_tokens=total,
                         prefill_decode_logit_relative_l2=rel,
                         same_top1=bool((whole.argmax(-1)==out.logits.argmax(-1)).all()),
                         explicit_kernel_checks=list(rt.operator_checks), before=before, after=after))
        print(json.dumps({'event': 'verified', **rows[-1]}), flush=True)
        save(P / 'checks/correctness_partial.json', dict(passed=False, cases=rows))
        rt.reset()
    for case, rt in runtimes.items():
        for total in (272,8192):
            check(case, rt, total, 'bfloat16')
    # Keep the original BF16 parameter tensors and restore them exactly. Do not
    # cast rotary buffers through BF16 while changing the validation precision.
    original_parameters = [(param, param.data) for param in model.parameters()]
    try:
        for param, value in original_parameters:
            param.data = value.float()
        for case, rt in runtimes.items():
            for total in (272,8192):
                check(case, rt, total, 'float32')
    finally:
        for param, value in original_parameters:
            param.data = value
    result = dict(passed=True, cases=rows,
                  tolerance_note='Explicit kernel vs FP32 scan relative L2<2e-4; FP32 whole-model '
                                 'one-shot vs prefill+decode relative logits L2<1e-3 at 272 and8192 tokens. '
                                 'BF16 differences reported as diagnostics, not gated by a relaxed '
                                 'threshold: an initial BF16-only 0.03 gate failed on AD (0.05367) '
                                 'and teacher was already0.02658, motivating independent FP32 validation. '
                                 'Original BF16 parameters preserved exactly for actual timing.')
    save(P / 'checks/correctness.json', result)
    return result


@torch.inference_mode()
def trial(model, rt, tokens, length, steps):
    activate(rt)
    prompt = tokens[:, :length]
    continuation = [tokens[:, t:t+1] for t in range(length, length+steps)]
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    c, d = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    base_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    a.record()
    cache = cache_for(model, rt.case != 'teacher')
    out = forward(model, prompt, cache)
    b.record()
    b.synchronize()
    prefill_wall = (time.perf_counter()-t0)*1000
    peak = torch.cuda.max_memory_allocated()-base_memory
    before = validate_storage(cache, rt, length)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    c.record()
    for ids in continuation:
        out = forward(model, ids, cache)
    d.record()
    d.synchronize()
    decode_wall = (time.perf_counter()-t0)*1000
    assert torch.isfinite(out.logits).all()
    after = validate_storage(cache, rt, length+steps)
    assert before['state_bytes'] == after['state_bytes']
    assert all(torch.isfinite(t).all() for state in rt.states.values() for t in state.values())
    row = dict(case=rt.case, prompt_tokens=length, decode_tokens=steps,
               prefill_cuda_ms=a.elapsed_time(b), prefill_wall_ms=prefill_wall,
               decode_cuda_ms_per_token=c.elapsed_time(d)/steps,
               decode_wall_ms_per_token=decode_wall/steps,
               request_wall_ms=prefill_wall+decode_wall,
               prefill_incremental_peak_bytes=peak,
               cache_after_prefill=before, cache_after_decode=after)
    rt.reset()
    return row


@torch.inference_mode()
def benchmark(args, model, runtimes, token_sets, metadata):
    results, environment = [], []
    rng = random.Random(77331)
    schedule = {str(n): [rng.sample(list(CASES), len(CASES)) for _ in range(args.blocks)]
                for n in args.lengths}
    metadata.update(blocks_per_length=args.blocks, decode_tokens=args.steps,
                    lengths=args.lengths, schedule=schedule,
                    warmup='Each method: two prompt+8decode warmups at each length. '
                           'Three documents cycle by paired block; randomized method order within block. '
                           'All measured blocks retained; CUDA events and synchronized wall clock.')
    save(P / 'results/protocol.json', metadata)
    for length in args.lengths:
        gc.collect()
        torch.cuda.empty_cache()
        for case in CASES:
            for _ in range(2):
                trial(model, runtimes[case], token_sets[0], length, 8)
        print(json.dumps(dict(event='warmup_complete', length=length)), flush=True)
        for block, order in enumerate(schedule[str(length)]):
            snapshot = {'length': length, 'block': block, 'gpu': gpu_snapshot()}
            environment.append(snapshot)
            for order_index, case in enumerate(order):
                row = trial(model, runtimes[case], token_sets[block % 3], length, args.steps)
                row.update(block=block, document=block % 3, order_index=order_index)
                results.append(row)
                print(json.dumps(dict(event='timing', **row)), flush=True)
                save(P / 'results/raw.json', dict(metadata=metadata, results=results,
                                                 environment=environment))
    return results


def summarize():
    raw = json.loads((P / 'results/raw.json').read_text())
    rows = raw['results']
    aggregates, comparisons = [], []
    lengths = sorted({r['prompt_tokens'] for r in rows})
    for n in lengths:
        for case in CASES:
            group = [r for r in rows if r['prompt_tokens']==n and r['case']==case]
            metrics = ['prefill_wall_ms', 'decode_wall_ms_per_token', 'request_wall_ms',
                       'prefill_cuda_ms', 'decode_cuda_ms_per_token', 'prefill_incremental_peak_bytes']
            aggregates.append(dict(case=case, prompt_tokens=n, trials=len(group),
                                   **{key: statistics.median(r[key] for r in group) for key in metrics},
                                   cache_after_prefill=group[0]['cache_after_prefill'],
                                   decode_wall_min=min(r['decode_wall_ms_per_token'] for r in group),
                                   decode_wall_max=max(r['decode_wall_ms_per_token'] for r in group)))
        for acase in ['ad_plain', 'ad_matched']:
            for bcase in ['hh_exp', 'hh_softmax', 'teacher']:
                ag = {r['block']: r for r in rows if r['prompt_tokens']==n and r['case']==acase}
                bg = {r['block']: r for r in rows if r['prompt_tokens']==n and r['case']==bcase}
                blocks = sorted(ag.keys() & bg.keys())
                for metric in ['prefill_wall_ms', 'decode_wall_ms_per_token', 'request_wall_ms']:
                    av = torch.tensor([ag[b][metric] for b in blocks], dtype=torch.float64)
                    bv = torch.tensor([bg[b][metric] for b in blocks], dtype=torch.float64)
                    generator = torch.Generator().manual_seed(91337)
                    indices = torch.randint(len(blocks), (20000,len(blocks)), generator=generator)
                    # Pair bootstrap whole timing blocks, not the correlated tokens within them.
                    savings = 100*(1-av[indices].mean(1)/bv[indices].mean(1))
                    comparisons.append(dict(prompt_tokens=n, a=acase, b=bcase, metric=metric,
                        paired_blocks=len(blocks), mean_difference_a_minus_b=float((av-bv).mean()),
                        latency_reduction_percent=100*(1-float(av.mean()/bv.mean())),
                        bootstrap_95_percent=savings.quantile(torch.tensor([.025,.975],dtype=torch.float64)).tolist(),
                        a_faster_blocks=int((av<bv).sum())))
    result = dict(aggregates=aggregates, comparisons=comparisons,
                  uncertainty='Conditional timing repeatability only: paired-block bootstrap across '
                              '12 blocks/3 fixed prompts on one GPU. Not uncertainty over hardware '
                              'or model training seeds. Positive reduction means AD faster.')
    save(P / 'results/summary.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['verify', 'benchmark', 'summarize'])
    parser.add_argument('--lengths', type=int, nargs='+', default=[1024,4096,8192])
    parser.add_argument('--blocks', type=int, default=12)
    parser.add_argument('--steps', type=int, default=64)
    args = parser.parse_args()
    if args.action == 'summarize':
        summarize()
        return
    model, runtimes, token_sets, original, metadata = setup()
    try:
        verification = verify(model, runtimes, token_sets)
        save(P / 'checks/environment.json', metadata)
        if args.action == 'benchmark':
            benchmark(args, model, runtimes, token_sets, metadata)
            summarize()
    finally:
        modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa', original)


if __name__ == '__main__':
    main()
