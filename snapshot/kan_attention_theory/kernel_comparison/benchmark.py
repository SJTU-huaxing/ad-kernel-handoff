"""Single-GPU inference microbenchmark, no concurrent workloads.

Includes feature extraction and rectangular linear-attention aggregation.
This is not end-to-end LLM throughput and does not benchmark full causal prefill.
"""
import json,math,time
import torch
from rf import P,OP,RandomFeatures,save,SEEDS,METHODS,pooled_model
from kernels import GalerkinFeatures,PartitionFeatures,linear_attention
from run import DATA

def elapsed(fn,reps=20,warmup=4):
    for _ in range(warmup):fn()
    torch.cuda.synchronize()
    starts=[torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends=[torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for start,end in zip(starts,ends):
        start.record();fn();end.record()
    torch.cuda.synchronize()
    samples=sorted(s.elapsed_time(e) for s,e in zip(starts,ends))
    return dict(median_ms=(samples[reps//2-1]+samples[reps//2])/2,
                p10_ms=samples[reps//10],p90_ms=samples[-reps//10-1],repeats=reps)

def storage_bytes(model):
    storages={}
    for x in vars(model).values():
        if torch.is_tensor(x):storages[x.untyped_storage().data_ptr()]=x.untyped_storage().nbytes()
    return sum(storages.values())

def cost(method,m):
    d=128;a=1024;e=64;dv=128
    if method in ['favor_plus','centered_favor_plus','sderf']:
        feature=4*d*m;parameters=d*m;exps=2*m
    elif method=='aderf':
        feature=4*d*(d+m);parameters=2*d*m+2*d*d;exps=2*m
    elif method=='galerkin':
        feature=4*a*(d+m);parameters=2*a*(d+m);exps=2*a
    else:
        feature=4*a*(d+e)+4*m*(m-1);parameters=2*a*(d+e)+m*m;exps=2*a
    return dict(dominant_feature_flops_per_query_key_pair=feature,
                aggregation_flops_per_query_key_pair=4*m*dv,
                total_dominant_flops_per_query_key_pair=feature+4*m*dv,
                exponentials_per_query_key_pair=exps,
                principal_float_coefficients_per_head=parameters,
                recurrent_state_scalars_per_head=m*(dv+1))

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    manifest=json.loads((DATA/'manifest.json').read_text())
    row=next(r for r in manifest['shards'] if r['split']=='test');raw=torch.load(DATA/row['file'],weights_only=True)
    q0=raw['q'][:16,:,512:].permute(1,0,2,3).flatten(1,2).cuda()
    k0=raw['k'][:16,:,:512].permute(1,0,2,3).flatten(1,2).cuda()
    v0=raw['v'][:16,:,:512].permute(1,0,2,3).flatten(1,2).cuda()
    bank=torch.load(P/'results/rf_models.pt',weights_only=True)
    selected={r['method']:r for r in bank if r['seed']==SEEDS[0]}
    rows=[]
    for dtype in [torch.float32,torch.float64]:
        for m in [16,32,64,128,640]:
            for method in METHODS+['partition','galerkin']:
                if m==640 and method not in METHODS:continue
                rf_model=pooled_model(bank,method) if m==640 else selected.get(method)
                model=RandomFeatures(rf_model,m,dtype) if method in METHODS else PartitionFeatures(m,dtype) if method=='partition' else GalerkinFeatures(m,dtype)
                # A fixed head/side numerical gauge, not row normalization or fitted amplitude.
                # Raw kernel is recovered by restoring the two constants; attention is unchanged.
                if isinstance(model,RandomFeatures):
                    sq=model.log_features(q0,'q').amax((1,2));sk=model.log_features(k0,'k').amax((1,2))
                    model.bq=model.bq-sq[:,None];model.bk=model.bk-sk[:,None]
                for n in [512,2048,8192]:
                    q=q0[:,:n].to(dtype);k=k0[:,:n].to(dtype);v=v0[:,:n].to(dtype)
                    def feature():return model.features(q,'q'),model.features(k,'k')
                    fq,fk=feature()
                    def aggregate():return linear_attention(fq,fk,v)
                    def total():
                        f,g=feature();return linear_attention(f,g,v)
                    output,den=aggregate()
                    timing=dict(feature=elapsed(feature),aggregation=elapsed(aggregate),total=elapsed(total))
                    torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                    total();torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()-base
                    record=dict(method=method,m=m,n=n,heads=4,dtype=str(dtype).split('.')[-1],
                        timing=timing,incremental_peak_bytes=peak,model_tensor_storage_bytes=storage_bytes(model),
                        nonfinite_output_fraction=float((~torch.isfinite(output)).float().mean()),
                        nonpositive_denominator_fraction=float((den<=0).float().mean()),**cost(method,m))
                    rows.append(record)
                    save(P/'results/benchmark.json',dict(results=rows,device=torch.cuda.get_device_name(),
                        torch_version=str(torch.__version__),protocol='4 heads, batch=1, d=d_v=128; rectangular all-visible aggregation; no backward, TF32 disabled; 4 warmups and 20 CUDA-event measurements.',
                        precision='FP32 and FP64 separately. Fixed per-head/side RF gauges prevent avoidable overflow and cancel exactly in normalized attention.',
                        scope='Microbenchmark, not full LLM throughput. Inputs concatenate real cached positions; timing lengths do not assert long-context accuracy.',
                        flop_note='Dominant matmul multiply-add counts (2 FLOPs/MAC); excludes exponentials, comparisons, normalization and small elementwise operations. Partition routing includes its dense path-score GEMM.',
                        parameter_note='Principal theoretical coefficients exclude small biases and routing auxiliaries; actual model tensor storage is also recorded.'))
                    print(json.dumps(dict(event='benchmark',method=method,m=m,n=n,dtype=str(dtype),total_ms=timing['total']['median_ms'])),flush=True)
                del model

if __name__=='__main__':main()
