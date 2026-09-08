import statistics
from core_attribution import *
from operators import step

def slice_layer(net):
    for mod in net.modules():
        for key,value in list(mod._parameters.items()):
            if value is not None:mod._parameters[key]=nn.Parameter(value[:12].contiguous(),requires_grad=False)
        for key,value in list(mod._buffers.items()):
            if value is not None and value.ndim and value.shape[0]==24:mod._buffers[key]=value[:12].contiguous()
    net.heads=12;net.runtime_raw=True
    return net

@torch.inference_mode()
def bench(fn,count=32):
    for _ in range(8):fn()
    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(count):fn()
    graph.replay();torch.cuda.synchronize();values=[]
    for _ in range(7):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();graph.replay();b.record();b.synchronize();values.append(a.elapsed_time(b)*1000/count)
    return dict(microseconds=statistics.median(values),samples_us=values)

@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    ds=source.data('confirm_wiki');q=ds['q'][0,:12,:1].cuda().float()
    k=ds['k'][0,:2,:1].cuda().float();v=ds['v'][0,:2,:1].cuda().float()
    cases=[dict(group=g,name=names[0]) for g,names in all_groups().items()];rows=[]
    for repeat in range(2):
        for case in cases if repeat==0 else list(reversed(cases)):
            net=slice_layer(load_fit(case['name'],torch.float32)[0])
            def features():return net.log_feature(q,'q'),net.log_feature(k.repeat_interleave(6,0),'k')
            _,lk=features()
            state=dict(s=torch.zeros(12,net.m,128,device='cuda'),z=torch.zeros(12,net.m,device='cuda'),g=lk[:,0].clone())
            def together():
                lq,lk=features();return step(lq,lk,v,state,optimized=False)
            row=dict(**case,repeat=repeat,m=net.m,features=bench(features),features_and_state=bench(together),
                state_bytes_per_layer=sum(t.numel()*t.element_size() for t in state.values()))
            rows.append(row);print(json.dumps(row),flush=True);del net,state;torch.cuda.empty_cache()
    aggregates=[]
    for case in cases:
        chosen=[r for r in rows if r['group']==case['group']]
        aggregates.append(dict(**case,m=64,state_bytes_per_layer=chosen[0]['state_bytes_per_layer'],
            features_us=statistics.median([x for r in chosen for x in r['features']['samples_us']]),
            total_us=statistics.median([x for r in chosen for x in r['features_and_state']['samples_us']]),
            repeat_total_us=[r['features_and_state']['microseconds'] for r in chosen]))
    save(P/'results/benchmark.json',dict(rows=rows,aggregates=aggregates,gpu=torch.cuda.get_device_name(),
        projection_flops_per_head=147072,state_flops_per_head=32768,
        scope='FP32 batch1 single-layer12heads CUDA graph hot cache; same generic reference state operator and fresh reversed-order repeats. Not full-model decode, not fused best achievable performance. FLOPs exclude nonlinearities/normalization/rescaling.'))

if __name__=='__main__':main()
