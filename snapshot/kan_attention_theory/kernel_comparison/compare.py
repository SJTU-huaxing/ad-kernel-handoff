"""Exact Float64 full-product RF risk through sufficient moments.

The full kernel is streamed once per head/dataset; all models share those values.
No test-time feature fitting, node selection, or kernel clipping.
"""
import argparse, json, math, time
import torch
from rf import P,OP,HEADS,RANKS,load_data,build_models,bank_features_one_head,save

@torch.no_grad()
def evaluate(q,k,models,head,ds,scale,energy,block=4096):
    h=HEADS.index(head);n=q.shape[1];assert n==k.shape[1]
    fq,sq=bank_features_one_head(q[h],'q',models,h)
    fk,sk=bank_features_one_head(k[h],'k',models,h)
    count=len(models);cross=torch.zeros(count,128,device='cuda',dtype=torch.float64)
    gq=torch.zeros(count,128,128,device='cuda',dtype=torch.float64);gk=torch.zeros_like(gq)
    for a in range(0,n,block):
        aa=fq[a:a+block].reshape(-1,count,128).transpose(0,1)
        bb=fk[a:a+block].reshape(-1,count,128).transpose(0,1)
        gq+=aa.transpose(-1,-2)@aa;gk+=bb.transpose(-1,-2)@bb
    start=time.perf_counter();target_energy=torch.zeros((),device='cuda',dtype=torch.float64)
    for a in range(0,n,block):
        qq=q[h,a:a+block].double()
        applied=torch.zeros_like(fq[a:a+block])
        for b in range(0,n,block):
            target=(qq@k[h,b:b+block].double().T/math.sqrt(128)-scale).exp()
            target_energy+=target.square().sum()
            applied+=target@fk[b:b+block]
        cross+=(fq[a:a+block]*applied).reshape(-1,count,128).sum(0)
        print(json.dumps(dict(event='rf_product_scan',dataset=ds,head=head,rows=a+len(qq),n=n,seconds=time.perf_counter()-start)),flush=True)
    energy_error=abs(float(target_energy/energy-1))
    assert energy_error<1e-9,energy_error
    results=[]
    for i,model in enumerate(models):
        restore=sq[i]+sk[i]-scale
        for m in RANKS:
            # Stored bank uses 1/sqrt(128); prefix rank m uses 1/sqrt(m).
            logscale=restore+math.log(128/m)
            cross_sum=cross[i,:m].sum();pred2=(gq[i,:m,:m]*gk[i,:m,:m]).sum()
            log_cross=cross_sum.log()+logscale-energy.log()
            log_pred2=pred2.log()+2*logscale-energy.log()
            total=torch.logaddexp(torch.zeros_like(log_pred2),log_pred2)
            fraction=2*torch.exp(log_cross-total)
            assert fraction<1+1e-8
            log_risk=total+torch.log1p(-fraction.clamp_max(1-1e-15))
            value=float(log_risk)
            results.append(dict(dataset=ds,n=n,head=head,method=model['method'],seed=model['seed'],m=m,
                log_nmse=value,nmse=math.exp(value) if value<700 else None,
                log_cross_over_energy=float(log_cross),log_prediction_energy_ratio=float(log_pred2)))
    save(P/f'results/{ds}_L{head[0]}H{head[1]}.json',dict(results=results,pairs=n*n,
        seconds=time.perf_counter()-start,reference_energy_relative_error=energy_error,
        arithmetic='Float64 throughout. Per-model common feature scales exactly restored.'))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--datasets',nargs='+',default=['official','internal']);args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    data=load_data()
    modelpath=P/'results/rf_models.pt'
    models=torch.load(modelpath,weights_only=True) if modelpath.exists() else build_models(data)
    del data['calibration'],data['moment_train']
    scales=torch.load(OP/'results/train_basis.pt',weights_only=True)['scale'].cuda()
    for ds in args.datasets:
        ref=torch.load(OP/f'results/{ds}_moments.pt',weights_only=True)
        n=ref['sizes'][-1];energy=ref['accumulators'][str(n)]['energy'].cuda()
        for h,head in enumerate(HEADS):
            if (P/f'results/{ds}_L{head[0]}H{head[1]}.json').exists():continue
            evaluate(data[ds]['q'],data[ds]['k'],models,head,ds,scales[h],energy[h])

if __name__=='__main__':main()
