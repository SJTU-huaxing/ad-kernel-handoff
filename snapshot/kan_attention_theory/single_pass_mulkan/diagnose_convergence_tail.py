"""Read-only checkpoint inference: convergence evidence and large-kernel localization."""
import json
import math
from collections import defaultdict
from pathlib import Path
import torch
from validate import load_model

P=Path(__file__).resolve().parent
OUT=P/'diagnostics';OUT.mkdir(exist_ok=True)


def histories():
    rows=[]
    for loss in ['logcosh','poisson']:
        metric='logcosh' if loss=='logcosh' else 'normalized_idiv'
        for method in ['mlp','kan','mulkan']:
            runs=[json.loads(f.read_text()) for f in sorted((P/'fits').glob(f'{loss}_{method}_n4096_b1_*.json'))]
            curve=[]
            for step in [1,64,128,256,512,1024,2048,4096]:
                vals=torch.tensor([next(h for h in r['history'] if h['documents_seen']==step)['validation'][metric] for r in runs],dtype=torch.float64)
                curve.append(dict(step=step,mean=float(vals.mean()),seed_sd=float(vals.mean(1).std()),per_head=vals.mean(0).tolist()))
            rows.append(dict(loss=loss,method=method,metric=metric,curve=curve,
                relative_drop_2048_4096=1-curve[-1]['mean']/curve[-2]['mean']))
    return rows


@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    manifest=json.loads((P/'data/manifest.json').read_text())
    batches=[torch.load(P/'data'/s['file'],weights_only=True,map_location='cpu') for s in manifest['shards'] if s['split']=='test']
    full={side:torch.cat([b[side] for b in batches]) for side in ['q','k']};del batches
    gen=torch.Generator().manual_seed(20280906)
    perm=torch.stack([torch.randperm(512,generator=gen) for _ in range(256)])
    paired_q=full['q'][:,:,512:]
    paired_k=full['k'][:,:,:512].gather(2,perm[:,None,:,None].expand(-1,4,-1,128))
    q=paired_q.permute(1,0,2,3).flatten(1,2).cuda()
    k=paired_k.permute(1,0,2,3).flatten(1,2).cuda()
    z=(q.float()*k.float()).sum(-1)/math.sqrt(128)
    nq=q.float().norm(dim=-1);nk=k.float().norm(dim=-1)
    cosine=z*math.sqrt(128)/(nq*nk)
    z=z.double();truth=z.exp();energy=truth.square()
    n=z.shape[1];count=math.ceil(n*.001);tail=torch.zeros_like(z,dtype=torch.bool)
    threshold=[];rows=[];keypos=perm.flatten().cuda();docpos=torch.arange(256,device='cuda').repeat_interleave(512)
    querypos=torch.arange(512,1024,device='cuda').repeat(256)
    for h,label in enumerate(manifest['head_labels']):
        ids=z[h].topk(count).indices;tail[h,ids]=True;threshold.append(float(z[h,ids].min()))
        key_counts=torch.bincount(keypos[ids],minlength=512)
        docs=torch.bincount(docpos[ids],minlength=256)
        kp=keypos[ids];row=dict(head=label,count=count,n=n,
            tail_energy_share=float(energy[h,ids].sum()/energy[h].sum()),
            tail_mass_share=float(truth[h,ids].sum()/truth[h].sum()),
            energy_weight_concentration_count=float(energy[h].sum().square()/energy[h].square().sum()),
            mass_weight_concentration_count=float(truth[h].sum().square()/energy[h].sum()),
            top_single_pair_energy_share=float(energy[h].max()/energy[h].sum()),
            tail_documents=int((docs>0).sum()),top_document_tail_count=int(docs.max()),
            top_key_positions=[dict(position=int(i),count=int(key_counts[i])) for i in key_counts.topk(8).indices if key_counts[i]>0],
            largest_pairs=[dict(document=int(docpos[i]),query_position=int(querypos[i]),key_position=int(keypos[i]),
                logit=float(z[h,i]),q_norm=float(nq[h,i]),k_norm=float(nk[h,i]),cosine=float(cosine[h,i]),
                energy_share=float(energy[h,i]/energy[h].sum())) for i in z[h].topk(5).indices],
            tail_query_min=int(querypos[ids].min()),tail_query_max=int(querypos[ids].max()))
        for limit in [1,4,8,16]:
            mask=keypos<limit;rest=~mask;num=math.ceil(int(rest.sum())*.001)
            values=energy[h,rest];rest_tail=values.topk(num).values
            row[f'first{limit}_tail_count_share']=float((kp<limit).double().mean())
            row[f'first{limit}_energy_share']=float(energy[h,mask].sum()/energy[h].sum())
            row[f'exclude_first{limit}_top001_energy_share']=float(rest_tail.sum()/values.sum())
        for name,vals in [('q_norm',nq),('k_norm',nk),('cosine',cosine),('logit',z)]:
            row[name]=dict(all_mean=float(vals[h].mean()),tail_mean=float(vals[h,ids].mean()),
                           all_median=float(vals[h].median()),tail_median=float(vals[h,ids].median()))
        rows.append(row)
    # Assert identical pairing/statistic to the previously reported population.
    previous=json.loads((P/'distribution.json').read_text())
    for a,b in zip(rows,previous['test']):assert abs(a['tail_energy_share']-b['top_0.001_energy_share'])<1e-6
    result=dict(history=histories(),paired_tail=rows,
        note='All data are the previous internally held-out test documents. No optimization or parameter changes. '
             'Concentration counts summarize weights; they are not counts of statistically independent samples.')
    predictions=[]
    for loss in ['logcosh','poisson']:
        for method in ['mlp','kan','mulkan']:
            for seed in [11,29,47]:
                name=f'{loss}_{method}_n4096_b1_s{seed}'
                model,meta=load_model(P/'fits'/(name+'.pt'),device='cuda')
                pred=torch.cat([model.log_kernel_pairs(q[:,i:i+8192].float(),k[:,i:i+8192].float()).double()
                                for i in range(0,n,8192)],1)
                pred+=torch.tensor(meta['log_scale'],device='cuda',dtype=torch.float64)[:,None]
                ratio=pred-z;hat=pred.exp();sse=(hat-truth).square()
                per_head=[]
                for h in range(4):
                    per_head.append(dict(head=manifest['head_labels'][h],
                        raw_nmse=float(sse[h].sum()/energy[h].sum()),
                        tail_sse_share=float(sse[h,tail[h]].sum()/sse[h].sum()),
                        tail_nmse=float(sse[h,tail[h]].sum()/energy[h,tail[h]].sum()),
                        ordinary_nmse=float(sse[h,~tail[h]].sum()/energy[h,~tail[h]].sum()),
                        tail_median_prediction_ratio=float(ratio[h,tail[h]].median().exp()),
                        tail_mass_prediction_ratio=float(hat[h,tail[h]].sum()/truth[h,tail[h]].sum())))
                predictions.append(dict(loss=loss,method=method,seed=seed,per_head=per_head))
                print(json.dumps(dict(event='predictions',name=name)),flush=True)
                del model
    result['stratified_prediction']=predictions
    # Complete legal rectangular blocks, 32 docs. Per-row normalized numbers are diagnostics only.
    dense=[]
    for i in range(32):
        a=full['q'][i,:,512:].float().cuda();b=full['k'][i,:,:512].float().cuda()
        zz=(a@b.transpose(-1,-2)/math.sqrt(128)).double();ker=zz.exp();en=ker.square();attn=zz.softmax(-1)
        per_head=[]
        for h in range(4):
            ke=ker[h];ee=en[h];mask=zz[h]>=threshold[h]
            row=dict(head=manifest['head_labels'][h],mass=float(ke.sum()),energy=float(ee.sum()),
                same_threshold_count=int(mask.sum()),same_threshold_energy=float(ee[mask].sum()),
                same_threshold_average_attention_mass=float((attn[h]*mask).sum(-1).mean()),
                mean_row_max_attention=float(attn[h].amax(-1).mean()))
            for limit in [1,4,8]:
                row[f'first{limit}_energy']=float(ee[:,:limit].sum())
                row[f'first{limit}_mass']=float(ke[:,:limit].sum())
                row[f'first{limit}_mean_row_attention']=float(attn[h,:,:limit].sum(-1).mean())
            per_head.append(row)
        dense.append(dict(ordinal=i,per_head=per_head))
    result['dense_blocks']=dense
    (OUT/'convergence_tail.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(event='complete',output=str(OUT/'convergence_tail.json'))),flush=True)


if __name__=='__main__':main()
