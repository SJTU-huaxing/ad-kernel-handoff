"""m=640 RF budget reference using all five predefined m=128 node blocks.

Reuse exact cross terms by linearity; compute all cross-block feature Gram terms.
No second scan of the dense true kernel is required. One union realization,
not five independent m=640 seeds.
"""
import json,math
import torch
from rf import P,OP,METHODS,HEADS,load_data,pooled_model,bank_features_one_head,save

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    data=load_data();del data['calibration'],data['moment_train']
    models=torch.load(P/'results/rf_models.pt',weights_only=True)
    basis=torch.load(OP/'results/train_basis.pt',weights_only=True)
    results=[]
    for ds in ['internal','official']:
        ref=torch.load(OP/f'results/{ds}_moments.pt',weights_only=True);n=ref['sizes'][-1]
        for h,head in enumerate(HEADS):
            raw=json.loads((P/f'results/{ds}_L{head[0]}H{head[1]}.json').read_text())['results']
            energy=ref['accumulators'][str(n)]['energy'][h].cuda()
            for method in METHODS:
                model=pooled_model(models,method)
                fq,sq=bank_features_one_head(data[ds]['q'][h],'q',[model],h)
                fk,sk=bank_features_one_head(data[ds]['k'][h],'k',[model],h)
                gq=fq.T@fq;gk=fk.T@fk
                norm=(gq*gk).sum()
                log_norm=norm.log()+2*(sq[0]+sk[0]-basis['scale'][h].cuda())-energy.log()
                rr=[r for r in raw if r['method']==method and r['m']==128]
                log_cross=torch.logsumexp(torch.tensor([r['log_cross_over_energy'] for r in rr],device='cuda',dtype=torch.float64),0)-math.log(5)
                total=torch.logaddexp(torch.zeros_like(log_norm),log_norm)
                fraction=2*(log_cross-total).exp();assert fraction<1+1e-8
                risk=total+torch.log1p(-fraction.clamp_max(1-1e-15))
                results.append(dict(dataset=ds,n=n,head=head,method=method,m=640,
                    seed='predefined_union_of_five_blocks',nmse=float(risk.exp()),log_nmse=float(risk),
                    log_cross_over_energy=float(log_cross),log_prediction_energy_ratio=float(log_norm)))
                del fq,fk,gq,gk
                print(json.dumps(results[-1]),flush=True)
    save(P/'results/pooled_640.json',dict(results=results,
        note='One fixed union of all five original 128-node draws. Theoretical FLOPs include its larger attention state; no node/seed selection on test data.'))

if __name__=='__main__':main()
