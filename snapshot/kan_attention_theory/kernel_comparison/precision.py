"""Check FP32 vs FP64 kernel predictions on a fixed held-out 2048x2048 product.

This is a numerical precision diagnostic, not a repeat population evaluation.
"""
import json,math
import torch
from rf import P,OP,RandomFeatures,METHODS,SEEDS,save,pooled_model
from kernels import PartitionFeatures,GalerkinFeatures
from run import DATA,HEADS

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    manifest=json.loads((DATA/'manifest.json').read_text())
    rr=next(r for r in manifest['shards'] if r['split']=='test')
    raw=torch.load(DATA/rr['file'],weights_only=True)
    q=raw['q'][:4,:,512:].permute(1,0,2,3).flatten(1,2).cuda()
    k=raw['k'][:4,:,:512].permute(1,0,2,3).flatten(1,2).cuda()
    target=q.double()@k.double().transpose(-1,-2)/math.sqrt(128)
    scale=target.amax((1,2));target=(target-scale[:,None,None]).exp()
    energy=target.square().sum((1,2))
    bank=torch.load(P/'results/rf_models.pt',weights_only=True)
    selected={r['method']:r for r in bank if r['seed']==SEEDS[0]}
    rows=[];pooled_error=0.
    for m in [64,640]:
        for method in METHODS+(['partition','galerkin'] if m==64 else []):
            predictions=[];labels=[]
            entry=pooled_model(bank,method) if m==640 else selected.get(method)
            for dtype in [torch.float64,torch.float32]:
                model=RandomFeatures(entry,m,dtype) if method in METHODS else PartitionFeatures(m,dtype) if method=='partition' else GalerkinFeatures(m,dtype)
                if method in METHODS:
                    lq=model.log_features(q,'q');lk=model.log_features(k,'k')
                    if dtype==torch.float64:sq=lq.amax((1,2));sk=lk.amax((1,2))
                    fq=(lq-sq.to(dtype)[:,None,None]).exp();fk=(lk-sk.to(dtype)[:,None,None]).exp()
                    pred=(fq@fk.transpose(-1,-2)).double()*(sq+sk-scale).exp()[:,None,None]
                else:
                    fq=model.features(q,'q')/model.sqrt_scale[:,None,None]
                    fk=model.features(k,'k')/model.sqrt_scale[:,None,None]
                    pred=(fq@fk.transpose(-1,-2)).double()*(model.scale.double()-scale).exp()[:,None,None]
                    if method=='partition':labels.append((model.cells(q,'q'),model.cells(k,'k')))
                predictions.append(pred)
            ref,low=predictions
            errors=(low-ref).square().sum((1,2))/energy
            for h,head in enumerate(HEADS):
                rows.append(dict(method=method,m=m,head=head,
                    nmse64=float((ref[h]-target[h]).square().sum()/energy[h]),
                    nmse32=float((low[h]-target[h]).square().sum()/energy[h]),
                    prediction_difference_over_true_energy=float(errors[h]),
                    changed_partition_assignments=int((labels[0][0][h]!=labels[1][0][h]).sum()+(labels[0][1][h]!=labels[1][1][h]).sum()) if labels else None))
            # Independent direct verification that pooled features equal the five-kernel mean.
            if m==640:
                small_q=q[:,:32];small_k=k[:,:32];parts=[]
                for row in [r for r in bank if r['method']==method]:
                    f=RandomFeatures(row,128)
                    lf=f.log_features(small_q,'q')-sq[:,None,None]
                    lg=f.log_features(small_k,'k')-sk[:,None,None]
                    parts.append(lf.exp()@lg.exp().transpose(-1,-2))
                expected=torch.stack(parts).mean(0)*(sq+sk-scale).exp()[:,None,None]
                diff=((expected-ref[:,:32,:32]).square().sum()/expected.square().sum()).sqrt().item()
                pooled_error=max(pooled_error,diff);assert diff<1e-9
    save(P/'checks/precision.json',dict(results=rows,samples_per_marginal=2048,
        max_pooled_direct_prediction_relative_error=pooled_error,
        note='A fixed four-document product diagnoses arithmetic, not unknown-population FP32 accuracy. No TF32 or BF16.'))
    print(json.dumps(dict(status='passed',max_pooled_error=pooled_error,
        max_prediction_difference_over_true_energy=max(r['prediction_difference_over_true_energy'] for r in rows))),flush=True)

if __name__=='__main__':main()
