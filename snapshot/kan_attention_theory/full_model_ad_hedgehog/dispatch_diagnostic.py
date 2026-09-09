"""Compare host-dispatched and CUDA-graph attention component timing.

Run only after the main full-model benchmark, never concurrently on the GPU.
This diagnoses implementation overhead; it is not another full-model result.
"""
import json
import random
import statistics
import time

import torch
from benchmark import Runtime, CASES, ROOT, P, step, save


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ds = torch.load(ROOT / 'causal_direction/data/confirm_wiki_000.pt', weights_only=True)
    q = ds['q'][0, :12, :1].cuda().float().contiguous()
    k = ds['k'][0, :2, :1].cuda().float().contiguous()
    v = ds['v'][0, :2, :1].cuda().float().contiguous()
    del ds
    cases = [c for c in CASES if c != 'teacher']
    runtimes = {case: Runtime(None, case) for case in cases}
    rows = []
    rng = random.Random(83914)
    for block in range(6):
        for case in rng.sample(cases, len(cases)):
            net = runtimes[case].maps[14]
            lq = net.log_feature(q, 'q')
            lk = net.log_feature(k.repeat_interleave(6,0), 'k')
            state = dict(s=torch.zeros(12,net.m,128,device='cuda'),
                         z=torch.zeros(12,net.m,device='cuda'),g=lk[:,0].clone())
            def fn():
                a = net.log_feature(q, 'q')
                b = net.log_feature(k.repeat_interleave(6,0), 'k')
                return step(a,b,v,state,optimized=False)
            for _ in range(10): fn()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(64): fn()
            graph.replay()
            torch.cuda.synchronize()
            for mode in rng.sample(['host','graph'],2):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                start.record()
                if mode == 'host':
                    for _ in range(128): fn()
                    count = 128
                else:
                    graph.replay()
                    count = 64
                end.record()
                end.synchronize()
                rows.append(dict(case=case,block=block,mode=mode,
                                 cuda_us=start.elapsed_time(end)*1000/count,
                                 wall_us=(time.perf_counter()-t0)*1e6/count))
            del graph,state
    aggregates = []
    for case in cases:
        for mode in ['host','graph']:
            group = [r for r in rows if r['case']==case and r['mode']==mode]
            aggregates.append(dict(case=case,mode=mode,
                                   cuda_us=statistics.median(r['cuda_us'] for r in group),
                                   wall_us=statistics.median(r['wall_us'] for r in group)))
    output = dict(rows=rows,aggregates=aggregates,
                  scope='One layer/12heads, actual cached QKV, FP32, same feature+reference step. '
                        'Six shuffled-order blocks. Host dispatch128calls versus graph64calls. '
                        'Hot state/weights, fixed repeated token, not complete-model latency. '
                        'Difference includes dispatch, allocation and graph execution effects; '
                        'does not isolate Python alone or predict a compiled full-model speedup.')
    save(P / 'results/dispatch_diagnostic.json', output)
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
