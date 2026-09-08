"""Tail-error attribution and checks for the replicated analytic RF baselines."""
import math
import torch
from experiment import P, Features, save, json, a_opt
from evaluate import load_eval,extra_calibration

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    cal={k:v.cuda() for k,v in torch.load(P/'results/calibration.pt',weights_only=True).items()}
    tq,tk,aa=extra_calibration(cal)
    identity_error=(tq@tk.transpose(-1,-2)-cal['identity']).abs().max().item()
    assert identity_error<1e-7
    eig=cal['raw_sderf_eigenvalues'];a=a_opt(eig)
    normsum=sum(cal[s+'_cov'].diagonal(dim1=-2,dim2=-1).sum(-1)+cal[s+'_mean'].square().sum(-1)/math.sqrt(128) for s in ['q','k'])
    favor=2*eig.sum(-1)-normsum
    sderf=(torch.log1p(-4*a)-.5*torch.log1p(-8*a)+(1+1/(1-8*a))*eig).sum(-1)-normsum
    assert (sderf<=favor+1e-8).all()
    # The stochastic estimator's marginal unbiasedness and mean log-second-moment optimum
    # do not imply low expected squared error under the data distribution.
    algebra=dict(aderf_inverse_transform_error=identity_error,
        favor_train_mean_log_second_moment=favor.tolist(),sderf_train_mean_log_second_moment=sderf.tolist())
    pools,_=load_eval();models=[]
    for kind in ['shared','untied']:
        for seed in [11,29,47]:
            f=Features(cal,64,seed,learn=kind)
            f.load_state_dict(torch.load(P/f'fits/{kind}_m64_s{seed}.pt',weights_only=True)['state']);f.double()
            models.append((kind,seed,f))
    result=[];tail=[]
    for dataset,pool in pools.items():
        qp=pool['q'][:,:,512:].flatten(1,2);kp=pool['k'][:,:,:512].flatten(1,2)
        for rep in range(4):
            sample=json.loads((P/f'results/{dataset}_product_{rep}.json').read_text())['sampling']
            q=qp[:,sample['q_indices']].double();k=kp[:,sample['k_indices']].double()
            lt=q@k.transpose(-1,-2)/math.sqrt(128);energy=(2*lt).flatten(1).logsumexp(-1)
            topn=math.ceil(lt.shape[1]*lt.shape[2]*.001);ix=lt.flatten(1).topk(topn,dim=-1).indices
            high=lt.flatten(1).gather(1,ix)
            tail.append(dict(dataset=dataset,rep=rep,top_count=topn,pairs=lt.shape[1]*lt.shape[2],
                top_energy_fraction=((2*high).logsumexp(-1)-energy).exp().tolist(),
                energy_effective_pairs=(2*energy-(4*lt).flatten(1).logsumexp(-1)).exp().tolist()))
            for kind,seed,f in models:
                lq=f.log_feature(q,'q');lk=f.log_feature(k,'k')
                lp=torch.cat([(lq[:,b:b+32,None]+lk[:,None]).logsumexp(-1) for b in range(0,512,32)],1)
                r=lp-lt;logsse=2*(torch.maximum(lp,lt)+(-torch.expm1(-r.abs())).log())
                total=logsse.flatten(1).logsumexp(-1)
                above=logsse.masked_fill(r<=0,-torch.inf).flatten(1).logsumexp(-1)
                true_tail=logsse.flatten(1).gather(1,ix).logsumexp(-1)
                result.append(dict(dataset=dataset,rep=rep,kind=kind,seed=seed,
                    log_sse=total.tolist(),log_energy=energy.tolist(),overprediction_sse_fraction=(above-total).exp().tolist(),
                    true_top_01pct_sse_fraction=(true_tail-total).exp().tolist(),
                    prediction_to_target_mass_ratio=(lp.flatten(1).logsumexp(-1)-lt.flatten(1).logsumexp(-1)).exp().tolist(),
                    top_true_kernel_median_prediction_ratio=r.flatten(1).gather(1,ix).median(-1).values.exp().tolist()))
    save(P/'results/tail_diagnostics.json',dict(algebra_checks=algebra,target_tails=tail,learned_errors=result))
    print(json.dumps(algebra))

if __name__=='__main__':main()
