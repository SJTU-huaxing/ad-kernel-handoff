"""Raw exponential-kernel baselines: correct PRF norm factors and trained constant."""
import json
import math
from pathlib import Path
import torch


@torch.inference_mode()
def main():
    p=Path(__file__).resolve().parent;torch.set_num_threads(4)
    manifest=json.loads((p/'data_qwen25_1p5b/manifest.json').read_text());labels=[[14,0],[14,6],[27,0],[27,6]]
    idx=[manifest['head_labels'].index(label) for label in labels]
    checkpoint=torch.load(p/'raw_importance/raw_mlp_positive_m64_s11.pt',weights_only=True)
    scale=torch.tensor(checkpoint['metadata']['log_scale'],device='cuda',dtype=torch.float64)
    constant=checkpoint['state_dict']['amplitude'][:,0,0].cuda().double().square()
    normalizer=torch.load(p/'analysis_qwen25_1p5b/normalization.pt',weights_only=True)
    qmean=normalizer['q_mean'][idx].cuda().double();kmean=normalizer['k_mean'][idx].cuda().double()
    h,d,m=4,128,64;projections={}
    for seed in [11,29,47]:
        torch.manual_seed(seed)
        orthogonal,triangular=torch.linalg.qr(torch.randn(h,d,d,device='cuda',dtype=torch.float64))
        orthogonal*=torch.diagonal(triangular,dim1=-2,dim2=-1).sign()[:,None,:]
        radii=torch.randn(h,m,d,device='cuda',dtype=torch.float64).norm(dim=-1)
        projections[seed]=orthogonal[:,:,:m].transpose(-1,-2)*radii[:,:,None]
    results=[]
    docs=[r for r in manifest['documents'] if r['split'] in ['test','test_long','ood']]
    for n,doc in enumerate(docs):
        record=torch.load(p/'data_qwen25_1p5b'/doc['file'],weights_only=True)
        q,k=record['q'][idx,-512:].cuda().double(),record['k'][idx].cuda().double()
        pos=torch.arange(k.shape[1]-512,k.shape[1],device='cuda')
        mask=torch.arange(k.shape[1],device='cuda')[None,:]<=pos[:,None]
        target=(q@k.transpose(-1,-2)/math.sqrt(d)-scale[:,None,None]).masked_fill(~mask,-torch.inf).exp()
        energy=target.square().sum((-2,-1))
        pred=constant[:,None,None].expand_as(target)*mask
        results.append(dict(method='train_mean_constant',seed=None,split=doc['split'],document=doc['file'],
            error=(pred-target).square().sum((-2,-1)).tolist(),energy=energy.tolist()))
        for seed,w in projections.items():
            for centered in [False,True]:
                qx=q-qmean[:,None] if centered else q
                kx=k-kmean[:,None] if centered else k
                qlog=qx@w.transpose(-1,-2)*d**(-.25)-qx.square().sum(-1,keepdim=True)/(2*math.sqrt(d))
                klog=kx@w.transpose(-1,-2)*d**(-.25)-kx.square().sum(-1,keepdim=True)/(2*math.sqrt(d))
                if centered:
                    common=(qmean*kmean).sum(-1)/(2*math.sqrt(d))
                    qlog+=((qx*kmean[:,None]).sum(-1)/math.sqrt(d)+common[:,None])[:,:,None]
                    klog+=((kx*qmean[:,None]).sum(-1)/math.sqrt(d)+common[:,None])[:,:,None]
                err=torch.zeros(h,device='cuda',dtype=torch.float64)
                for start in range(0,512,32):
                    lp=(qlog[:,start:start+32,None,:]+klog[:,None,:,:]).logsumexp(-1)-math.log(m)-scale[:,None,None]
                    prediction=lp.masked_fill(~mask[None,start:start+32,:],-torch.inf).exp()
                    err+=(prediction-target[:,start:start+32]).square().sum((-2,-1))
                if not torch.isfinite(err).all():raise RuntimeError('Nonfinite random-feature raw error')
                results.append(dict(method='mean_centered_positive_ORF' if centered else 'positive_orthogonal_random_features',
                                    seed=seed,split=doc['split'],document=doc['file'],error=err.tolist(),energy=energy.tolist()))
        print(json.dumps(dict(event='raw_baselines',document=n+1,total=len(docs))),flush=True)
    (p/'raw_baselines.json').write_text(json.dumps(dict(head_labels=labels,m=m,log_scale=scale.tolist(),
        definition='phi(x)=exp(d^(-1/4)*W*x - ||x||^2/(2*sqrt(d)))/sqrt(m), same Gaussian ORF W for Q/K',
        constant_scaled_kernel=constant.tolist(),evaluations=results),indent=2))


if __name__=='__main__':main()
