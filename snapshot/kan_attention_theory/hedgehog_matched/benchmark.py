import copy
import statistics

from assess import *
from operators import step


@torch.inference_mode()
def bench(fn, count=32):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(count):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(7):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / count)
    return dict(microseconds=statistics.median(times), samples_us=times)


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    ds = source.data('confirm_wiki')
    q = ds['q'][0, :12, :1].cuda().float()
    k = ds['k'][0, :2, :1].cuda().float()
    v = ds['v'][0, :2, :1].cuda().float()
    cases = [n for n in plan()['models'] if '_s11_' in n]
    rows = []
    for name in cases + ['favor_s11']:
        if name == 'favor_s11':
            net, meta = source.load(name, torch.float32)
            for mod in net.modules():
                for key, value in list(mod._parameters.items()):
                    if value is not None:
                        mod._parameters[key] = nn.Parameter(value[:12].contiguous(), requires_grad=False)
                for key, value in list(mod._buffers.items()):
                    if value is not None and value.ndim and value.shape[0] == 24:
                        mod._buffers[key] = value[:12].contiguous()
            net.heads = 12
        else:
            net = Replacement(None, name).maps[14]
        def features():
            return net.log_feature(q, 'q'), net.log_feature(k.repeat_interleave(6, 0), 'k')
        lq, lk = features()
        m = net.m
        state = dict(s=torch.zeros(12, m, 128, device='cuda'),
                     z=torch.zeros(12, m, device='cuda'),
                     g=lk[:, 0].clone())
        def together():
            a, b = features()
            return step(a, b, v, state, optimized=False)
        feats = bench(features)
        total = bench(together)
        state_bytes = sum(t.numel()*t.element_size() for t in state.values())
        rows.append(dict(name=name, m=m, features=feats, features_and_state=total,
                         state_bytes_per_layer=state_bytes))
        print(json.dumps(rows[-1]), flush=True)
        del net, state
        torch.cuda.empty_cache()
    save(P/'results/benchmark.json', dict(
        rows=rows, gpu=torch.cuda.get_device_name(),
        scope='Batch1,12heads one layer,FP32,real heldout Q/K/V,CUDA graph hot-cache; '
              'same PyTorch reference projection and dynamic-state operations for all methods; '
              'not fused optimal kernels and not full-model latency.',
        arithmetic=dict(ad_projection_flops=147456, hh_projection_flops=147456,
                        ad_state_flops=32768, hh_state_flops=294912,
                        state_ratio=9, total_hh_to_ad_flops=442368/180224)))


if __name__ == '__main__':
    main()
