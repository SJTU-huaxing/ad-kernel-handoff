"""Frozen kernel/attention diagnostics on actual same-document rectangles.

Every query in positions 512:1024 can see all keys in positions 0:512.
This is an auxiliary 512-key attention block, not full causal LLM replacement.
"""
import json,math
import torch
from rf import P,OP,RandomFeatures,save,HEADS,METHODS,pooled_model
from kernels import GalerkinFeatures,PartitionFeatures
from run import DATA

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    bank=torch.load(P/'results/rf_models.pt',weights_only=True)
    models=[(dict(method=r['method'],seed=r['seed']),RandomFeatures(r,64)) for r in bank]
    models += [(dict(method='partition',seed=None),PartitionFeatures(64)),(dict(method='galerkin',seed=None),GalerkinFeatures(64))]
    models += [(dict(method=method,seed='pooled_5_blocks'),RandomFeatures(pooled_model(bank,method),640)) for method in METHODS]
    scale=torch.load(OP/'results/train_basis.pt',weights_only=True)['scale'].cuda()
    manifests={}
    manifest=json.loads((DATA/'manifest.json').read_text())
    manifests['internal']=[(DATA/r['file'],None) for r in manifest['shards'] if r['split']=='test']
    old=OP.parent/'real_llm_pilot/data_qwen25_1p5b';om=json.loads((old/'manifest.json').read_text())
    hi=[om['head_labels'].index(h) for h in HEADS]
    manifests['official']=[(old/r['file'],hi) for r in om['documents'] if r['split']=='test']
    for ds,files in manifests.items():
        outfile=P/f'results/{ds}_same_document.json'
        if outfile.exists():continue
        sums=[torch.zeros(11,4,device='cuda',dtype=torch.float64) for _ in models]
        docs=0
        for path,index in files:
            raw=torch.load(path,weights_only=True)
            if index is not None:raw={s:raw[s][index][None] for s in ['q','k','v']}
            for a in range(0,len(raw['q']),4):
                q=raw['q'][a:a+4,:,512:].permute(1,0,2,3).cuda()
                k=raw['k'][a:a+4,:,:512].permute(1,0,2,3).cuda()
                v=raw['v'][a:a+4,:,:512].permute(1,0,2,3).cuda().double()
                b=q.shape[1];docs+=b
                logits=q.double()@k.double().transpose(-1,-2)/math.sqrt(128)
                true=(logits-scale[:,None,None,None]).exp()
                probabilities=logits.softmax(-1);out=probabilities@v
                h=true.square().sum((1,2,3));den_true=true.sum(-1)
                for j,(meta,model) in enumerate(models):
                    m=model.m
                    if isinstance(model,RandomFeatures):
                        lq=model.log_features(q.flatten(1,2),'q');lk=model.log_features(k.flatten(1,2),'k')
                        sq=lq.amax((1,2));sk=lk.amax((1,2))
                        fq=(lq-sq[:,None,None]).exp().reshape(4,b,512,m)
                        fk=(lk-sk[:,None,None]).exp().reshape(4,b,512,m)
                        restore=(sq+sk-scale).exp()[:,None,None,None]
                    else:
                        fq=(model.features(q.flatten(1,2),'q')/model.sqrt_scale[:,None,None]).reshape(4,b,512,m)
                        fk=(model.features(k.flatten(1,2),'k')/model.sqrt_scale[:,None,None]).reshape(4,b,512,m)
                        restore=1.
                    mat=fq@fk.transpose(-1,-2)
                    pred=mat*restore;den=mat.sum(-1)
                    weights=mat/den[:,:,:,None]
                    output=weights@v
                    # Check the actual linear-attention factorization independently.
                    den_linear=(fq*fk.sum(2)[:,:,None]).sum(-1)
                    output_linear=(fq@(fk.transpose(-1,-2)@v))/den_linear[:,:,:,None]
                    if torch.isfinite(output).all() and torch.isfinite(output_linear).all():
                        relative=((output-output_linear).square().sum()/output.square().sum().clamp_min(1e-300)).sqrt()
                        assert relative<1e-7,relative
                    invalid=~torch.isfinite(output).all(-1)
                    output_diff=(output-out).square().sum(-1)
                    output_diff[invalid]=0 # invalid rows counted explicitly; result then invalidated below
                    l1=(weights-probabilities).abs().sum(-1);l1[invalid]=0
                    den_pred=pred.sum(-1)
                    sums[j]+=torch.stack([
                        (pred-true).square().sum((1,2,3)),h,
                        output_diff.sum((1,2)),out.square().sum((1,2,3)),
                        (mat<0).double().sum((1,2,3)),(den<=0).double().sum((1,2)),
                        invalid.double().sum((1,2)),l1.sum((1,2)),
                        (den_pred-den_true).square().sum((1,2)),den_true.square().sum((1,2)),
                        (den_linear.abs()<1e-6*(fq.abs()*fk.abs().sum(2)[:,:,None]).sum(-1)).double().sum((1,2))])
                print(json.dumps(dict(event='same_document',dataset=ds,documents=docs)),flush=True)
        rows=[]
        for (meta,model),values in zip(models,sums):
            values=values.cpu()
            for h,head in enumerate(HEADS):
                x=values[:,h];valid=x[6]==0
                rows.append(dict(**meta,head=head,m=model.m,documents=docs,kernel_nmse=float(x[0]/x[1]),
                    output_nmse=float(x[2]/x[3]) if valid else None,
                    negative_kernel_fraction=float(x[4]/(docs*512**2)),nonpositive_denominator_fraction=float(x[5]/(docs*512)),
                    invalid_output_rows=int(x[6]),row_l1=float(x[7]/(docs*512)) if valid else None,
                    denominator_nmse=float(x[8]/x[9]),near_cancellation_fraction=float(x[10]/(docs*512))))
        save(outfile,dict(results=rows,protocol='All collected same-document 512-query x 512-key rectangles, frozen features; no clipping or denominator repair.',
            note='Auxiliary normalized outputs only. Kernel target and parameter construction remain unnormalized exp(q.k/sqrt(d)). Not full LLM perplexity.'))

if __name__=='__main__':main()
