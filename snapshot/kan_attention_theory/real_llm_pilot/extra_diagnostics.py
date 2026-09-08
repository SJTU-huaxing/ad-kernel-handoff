"""Remove sink positions, evaluate simple baselines and stable positive random features."""
import argparse
import json
import math
from pathlib import Path

import torch
from analyze_distribution import spectrum


def metrics(a, ahat, v):
    y, yh = a @ v, ahat @ v
    return dict(attention_nmse=((ahat-a).square().sum((-2,-1))/a.square().sum((-2,-1))).tolist(),
                output_nmse=((yh-y).square().sum((-2,-1))/y.square().sum((-2,-1)).clamp_min(1e-30)).tolist(),
                row_l1=(ahat-a).abs().sum(-1).mean(-1).tolist())


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    manifest=json.loads((args.data/'manifest.json').read_text());labels=manifest['head_labels']
    docs=[x for x in manifest['documents'] if x['split']=='test'][:12]
    records=[]
    for n,doc in enumerate(docs):
        d=torch.load(args.data/doc['file'],weights_only=True)
        q,k=d['q'][:,-512:].cuda().double(),d['k'][:,:512].cuda().double()
        full=q@k.transpose(-1,-2)/math.sqrt(q.shape[-1]);full_a=full.softmax(-1)
        for skip in [1,4,16]:
            specs,extra=spectrum(full[:,:,skip:],[16,32,64,128,256])
            for rows in specs:
                for h,row in enumerate(rows):
                    row.pop('singular_values')
                    row.update(document=doc['file'],layer=labels[h][0],head=labels[h][1],
                               removed_prefix=skip,remaining_keys=512-skip,
                               removed_original_mass=float(full_a[h,:,:skip].sum(-1).mean()),
                               **{key:float(value[h]) for key,value in extra.items()})
                    records.append(row)
        print(json.dumps(dict(event='sink_spectra',document=n+1,total=len(docs))),flush=True)
    (args.out/'sink_spectra.json').write_text(json.dumps(records,indent=2))

    selected=[[l,h] for l in [0,14,27] for h in [0,6]]
    indices=[labels.index(x) for x in selected];h=len(indices);dim=manifest['head_dimension'];m=64
    projections={}
    for seed in [11,29,47]:
        torch.manual_seed(seed)
        gaussian=torch.randn(h,dim,dim,device='cuda',dtype=torch.float64)
        orthogonal,triangular=torch.linalg.qr(gaussian)
        orthogonal=orthogonal*torch.diagonal(triangular,dim1=-2,dim2=-1).sign()[:,None,:]
        radii=torch.randn(h,m,dim,device='cuda',dtype=torch.float64).norm(dim=-1)
        projections[seed]=orthogonal[:,:,:m].transpose(-1,-2)*radii[:,:,None]
    evaluations=[]
    docs=[x for x in manifest['documents'] if x['split'] in ['test','test_long','ood']]
    for n,doc in enumerate(docs):
        d=torch.load(args.data/doc['file'],weights_only=True)
        q,k,v=[d[key][indices].cuda().double() for key in ['q','k','v']]
        positions=torch.arange(len(q[0])-512,len(q[0]),device='cuda');q=q[:,positions]
        mask=torch.arange(len(k[0]),device='cuda')[None,:]<=positions[:,None]
        a=(q@k.transpose(-1,-2)/math.sqrt(dim)).masked_fill(~mask,-torch.inf).softmax(-1)
        uniform=mask.double()[None].expand(h,-1,-1)/mask.sum(-1)[None,:,None]
        first=torch.zeros_like(a);first[:,:,0]=1
        for name,ahat in [('uniform',uniform),('first_token',first)]:
            evaluations.append(dict(method=name,seed=None,split=doc['split'],document=doc['file'],**metrics(a,ahat,v)))
        # Reference evaluation in log space. It deliberately does NOT benchmark a fused linear scan.
        for seed,w in projections.items():
            qlog=q@w.transpose(-1,-2)*dim**(-.25)
            klog=k@w.transpose(-1,-2)*dim**(-.25)-k.square().sum(-1,keepdim=True)/(2*math.sqrt(dim))
            parts=[]
            for begin in range(0,512,32):
                logkernel=(qlog[:,begin:begin+32,None,:]+klog[:,None,:,:]).logsumexp(-1)
                parts.append(logkernel.masked_fill(~mask[None,begin:begin+32,:],-torch.inf).softmax(-1))
            ahat=torch.cat(parts,1)
            evaluations.append(dict(method='positive_orthogonal_random_features',seed=seed,
                                    split=doc['split'],document=doc['file'],**metrics(a,ahat,v)))
        print(json.dumps(dict(event='baselines',document=n+1,total=len(docs))),flush=True)
    (args.out/'baselines.json').write_text(json.dumps(dict(head_labels=selected,m=m,evaluations=evaluations,
        note='Positive orthogonal Gaussian random features; query-only norm cancels in normalization; log-space reference, not optimized FAVOR+ implementation.'),indent=2))


if __name__=='__main__':main()
