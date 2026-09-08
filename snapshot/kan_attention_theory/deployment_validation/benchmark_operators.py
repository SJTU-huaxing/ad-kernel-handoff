"""Eager and CUDA-graph measurements, both kernels get equivalent optimization."""
import json,statistics
import torch
from runtime import P,FeatureMap,causal_prefill,recurrent_step

def timing(fn,graph=False):
    for _ in range(4):out=fn()
    torch.cuda.synchronize()
    if graph:
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):out=fn()
        fn=g.replay
    for _ in range(4):fn()
    samples=[]
    for _ in range(30):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        a.record();fn();b.record();b.synchronize();samples.append(a.elapsed_time(b))
    return dict(median_ms=statistics.median(samples),min_ms=min(samples),max_ms=max(samples))

@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    raw=torch.load(P.parent/'single_pass_mulkan/data/test_000.pt',weights_only=True)
    q0,k0,v0=[raw[s][:8].permute(1,0,2,3).flatten(1,2).cuda().float() for s in ['q','k','v']]
    rows=[];checks=[]
    for h in [2,4]:
        for method in ['favor_plus','partition','galerkin']:
            models=[FeatureMap(method,list(range(h)),optimized=b) for b in [False,True]]
            for n in [1,128,2048,8192]:
                q=q0[:h,:n].contiguous();k=k0[:h,:n].contiguous();v=v0[:h,:n].contiguous()
                reference=[models[0].raw_features(q,'q'),models[0].raw_features(k,'k')]
                for optimized,fmap in zip([False,True],models):
                    def feature():return fmap.raw_features(q,'q'),fmap.raw_features(k,'k')
                    fq,fk=feature()
                    err=max(float((a-b).norm()/b.norm().clamp_min(1e-30)) for a,b in zip([fq,fk],reference))
                    if method=='partition':
                        cellerr=max(float((models[0].cells(x,side)!=fmap.cells(x,side)).float().mean()) for x,side in [(q,'q'),(k,'k')])
                        assert cellerr==0
                    assert err<2e-4,(method,n,err)
                    checks.append(dict(method=method,heads=h,n=n,optimized=optimized,relative_l2=err))
                    for graph in [False,True]:
                        measured=timing(feature,graph)
                        row=dict(method=method,heads=h,n=n,optimized=optimized,cuda_graph=graph,feature_pair=measured)
                        rows.append(row)
                print(json.dumps(dict(event='operators',method=method,h=h,n=n)),flush=True)
    (P/'results/operator_benchmark.json').write_text(json.dumps(dict(results=rows,checks=checks,
        device=torch.cuda.get_device_name(),protocol='FP32, TF32 disabled, m=64,d=128,a=1024. Feature Q/K pair only, real cached Q/K. 4 warmups then 30 CUDA event repeats; eager and CUDA graph replay. No other GPU jobs.',
        optimization='Partition: fused exp epilogue, GEMV+exp for N=1, tree traversal instead of dense path-score GEMM. FAVOR+: fused norm/bias/exp and GEMV+norm/exp for N=1. Galerkin: same landmark epilogue/GEMV as partition. Main large GEMMs use cuBLAS.',
        limitation='Prototype implementations, no autotuning, no claim to best achievable kernels. CUDA graphs isolate much of host launch overhead; they do not remove mathematical FLOPs.'),indent=2))

if __name__=='__main__':main()
