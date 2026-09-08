"""Same PyTorch/CUDA-graph timing protocol for candidate feature pairs."""
import argparse,statistics,time
from common import *

def timeit(fn,graph):
    for _ in range(4):out=fn()
    torch.cuda.synchronize()
    if graph:
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):out=fn()
        fn=g.replay
    for _ in range(4):fn()
    values=[]
    for _ in range(20):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        a.record();fn();b.record();b.synchronize();values.append(a.elapsed_time(b))
    return dict(median_ms=statistics.median(values),samples_ms=values)

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--methods',nargs='*');args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    raw=torch.load(SINGLE/'data/test_000.pt',weights_only=True)
    q,k=[raw[s][:8].permute(1,0,2,3).flatten(1,2).cuda().float() for s in ['q','k']]
    methods=['favor_1009','favor_640','partition','galerkin','nystrom','spectral_pair','avg_anchor','cone_anchor',
        'avg_mlp','cone_mlp','avg_mulkan','cone_mulkan','nn_mlp_11','nn_kan_11','nn_mulkan_11','vq']
    if args.methods:methods=args.methods
    path=P/'results/feature_benchmark.json'
    rows=[r for r in json.loads(path.read_text())['results'] if r['method'] not in methods] if path.exists() else []
    checks=[]
    for name in methods:
        model=Candidate(name,torch.float32)
        for n in [1,128,2048,8192]:
            qq=q[:,:n].contiguous();kk=k[:,:n].contiguous()
            def feature():return model.features(qq,'q'),model.features(kk,'k')
            for graph in [False,True]:
                measured=timeit(feature,graph);rows.append(dict(method=name,n=n,heads=4,cuda_graph=graph,**measured))
            print(json.dumps(dict(event='feature_benchmark',method=name,n=n,eager_ms=rows[-2]['median_ms'],graph_ms=rows[-1]['median_ms'])),flush=True)
        del model
        save(P/'results/feature_benchmark.json',dict(results=rows,hardware=torch.cuda.get_device_name(),
            protocol='Identical PyTorch feature implementations, FP32 TF32 off, 4 heads, 4 warmups then 20 CUDA event repeats. Both eager and CUDA graph. No concurrent GPU jobs.',
            limitation='Prototype feature generation; no newly optimized custom kernels. The old separately optimized FAVOR+/partition timings are reported as an additional reference, not mixed into this table.'))

if __name__=='__main__':main()
