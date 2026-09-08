import statistics
from ablation import *
from operators import step


def slice_layer(net):
    for mod in net.modules():
        for key, value in list(mod._parameters.items()):
            if value is not None:
                mod._parameters[key] = nn.Parameter(value[:12].contiguous(), requires_grad=False)
        for key, value in list(mod._buffers.items()):
            if value is not None and value.ndim and value.shape[0] == 24:
                mod._buffers[key] = value[:12].contiguous()
    net.heads = 12
    return net


@torch.inference_mode()
def bench(fn, count=32):
    for _ in range(8): fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(count): fn()
    graph.replay(); torch.cuda.synchronize()
    values = []
    for _ in range(7):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); graph.replay(); b.record(); b.synchronize()
        values.append(a.elapsed_time(b)*1000/count)
    return dict(microseconds=statistics.median(values), samples_us=values)


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    ds = source.data('confirm_wiki')
    q = ds['q'][0,:12,:1].cuda().float()
    k = ds['k'][0,:2,:1].cuda().float()
    v = ds['v'][0,:2,:1].cuda().float()
    cases = []
    for regime in ['product_i','causal_kl']:
        for variant in VARIANTS:
            lr = plan()['learning_rates'][f'{regime}__{variant}']
            cases.append(dict(regime=regime,variant=variant,name=f'{regime}_{variant}_s11_lr{lr:g}',origin='new'))
        for kind in ['ad','hh']:
            cases.append(dict(regime=regime,variant=kind+'_original',name=old_name(regime,kind,11),origin='old'))
        cases.append(dict(regime=regime,variant='ad_inference_cancelled',name=old_name(regime,'ad',11),origin='cancelled'))
    cases.append(dict(regime='causal_kl',variant='hh_softmax',name='hh_softmax_kl_s11_lr0.002',origin='old'))
    cases.append(dict(regime='reference',variant='favor',name='favor_s11',origin='favor'))
    rows = []
    for repeat in range(2):
        order = cases if repeat == 0 else list(reversed(cases))
        for case in order:
            if case['origin'] == 'new':
                net, meta = load_fit(case['name'],torch.float32)
                net.runtime_raw = True
            elif case['origin'] == 'favor':
                net, meta = source.load(case['name'],torch.float32)
            else:
                net, meta = parent.load_fit(case['name'],torch.float32)
                if case['origin'] == 'cancelled':
                    net = inference_cancelled(net)
            net = slice_layer(net)
            def features():
                return net.log_feature(q,'q'),net.log_feature(k.repeat_interleave(6,0),'k')
            _, lk = features()
            state = dict(s=torch.zeros(12,net.m,128,device='cuda'),z=torch.zeros(12,net.m,device='cuda'),g=lk[:,0].clone())
            def together():
                lq, lk = features()
                return step(lq,lk,v,state,optimized=False)
            f, total = bench(features), bench(together)
            row = dict(**case,repeat=repeat,m=net.m,features=f,features_and_state=total,
                       state_bytes_per_layer=sum(t.numel()*t.element_size() for t in state.values()))
            rows.append(row)
            print(json.dumps(row),flush=True)
            del net,state
            torch.cuda.empty_cache()
    aggregates = []
    for case in cases:
        chosen = [r for r in rows if r['name']==case['name'] and r['variant']==case['variant']]
        aggregates.append(dict(**case,m=chosen[0]['m'],state_bytes_per_layer=chosen[0]['state_bytes_per_layer'],
            features_us=statistics.median([x for r in chosen for x in r['features']['samples_us']]),
            total_us=statistics.median([x for r in chosen for x in r['features_and_state']['samples_us']]),
            repeat_total_us=[r['features_and_state']['microseconds'] for r in chosen]))
    save(P/'results'/'benchmark.json',dict(rows=rows,aggregates=aggregates,gpu=torch.cuda.get_device_name(),
        scope='Fresh two-order same-hardware reference microbenchmark: FP32,batch1,12heads,real QKV,CUDA graph hot-cache. Not full-model decode or optimally fused kernels.',
        arithmetic=dict(original_ad_projection=147456,plain_projection=147072,matched_projection_including_hidden_and_direction_bias_i=147265,
                        matched_projection_including_hidden_bias_kl=147264,hh_projection=147456,
                        state_m64=32768,state_m576=294912,
                        scope='Dominant multiply/add counts; excludes nonlinearities, normalizations, numerical rescaling, and common LLM layers.')))


if __name__ == '__main__':main()
