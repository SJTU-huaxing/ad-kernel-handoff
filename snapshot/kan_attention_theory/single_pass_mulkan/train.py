"""Single-epoch, equal-parameter kernel learning. No raw-kernel MSE training."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import torch
from models import FeaturePair,loss_per_pair

P=Path(__file__).resolve().parent


def flatten_docs(x):return x.permute(1,0,2,3).flatten(1,2)
def flatten_logits(x):return x.permute(1,0,2).flatten(1,2)


@torch.no_grad()
def load_data():
    manifest=json.loads((P/'data/manifest.json').read_text());data={}
    for split in ['train','validation','test']:
        batches=[torch.load(P/'data'/r['file'],weights_only=True) for r in manifest['shards'] if r['split']==split]
        full={key:torch.cat([b[key] for b in batches]).cuda() for key in ['q','k']+([] if split=='train' else ['v'])}
        if split=='train':pairs=full
        else:
            generator=torch.Generator().manual_seed(20260906+(10000 if split=='validation' else 20000))
            permutations=torch.stack([torch.randperm(512,generator=generator) for _ in range(len(full['q']))]).cuda()
            keys=full['k'][:,:,:512].gather(2,permutations[:,None,:,None].expand(-1,4,-1,128))
            pairs=dict(q=full['q'][:,:,512:],k=keys)
        logits=[]
        for b in range(0,len(pairs['q']),32):
            logits.append((pairs['q'][b:b+32].float()*pairs['k'][b:b+32].float()).sum(-1)/math.sqrt(128))
        data[split]=dict(**pairs,logits=torch.cat(logits),full=full if split!='train' else None)
        del batches
    print(json.dumps(dict(event='loaded',documents={s:len(d['q']) for s,d in data.items()},gpu=torch.cuda.get_device_name())),flush=True)
    return manifest,data


@torch.no_grad()
def calibration(data,n):
    norm={}
    for side in ['q','k']:
        sample=flatten_docs(data[side][:n,:,::32]).double()
        norm[side+'_mean']=sample.mean(1).float()
        norm[side+'_std']=sample.std(1).clamp_min(.03).float()
    logits=flatten_logits(data['logits'][:n]).double()
    scale=(logits.logsumexp(-1)-math.log(logits.shape[1])).float()
    return norm,scale


@torch.no_grad()
def evaluate_pairs(model,data,scale,n=None,keep_records=True):
    n=n or len(data['q']);rows=[]
    for start in range(0,n,32):
        stop=min(start+32,n)
        q,k=[flatten_docs(data[side][start:stop]).float() for side in ['q','k']]
        pred=model.log_kernel_pairs(q,k).double().reshape(4,stop-start,512).permute(1,0,2)
        target=data['logits'][start:stop].double()-scale.double()[None,:,None]
        r=pred-target;hat=pred.exp();truth=target.exp()
        values=dict(logcosh=(r.abs()+torch.nn.functional.softplus(-2*r.abs())-math.log(2)).mean(-1),
                    log_mae=r.abs().mean(-1),factor2=(r.abs()<=math.log(2)).double().mean(-1),
                    factor10=(r.abs()<=math.log(10)).double().mean(-1),
                    raw_sse=(hat-truth).square().sum(-1),raw_energy=truth.square().sum(-1),
                    idiv=(hat-truth-truth*r).sum(-1),target_mass=truth.sum(-1))
        vals={key:value.cpu().tolist() for key,value in values.items()}
        for b in range(stop-start):rows.append(dict(ordinal=start+b,**{key:value[b] for key,value in vals.items()}))
    t={key:torch.tensor([r[key] for r in rows],dtype=torch.float64) for key in rows[0] if key!='ordinal'}
    summary={key:t[key].mean(0).tolist() for key in ['logcosh','log_mae','factor2','factor10']}
    summary['raw_nmse']=(t['raw_sse'].sum(0)/t['raw_energy'].sum(0)).tolist()
    summary['normalized_idiv']=(t['idiv'].sum(0)/t['target_mass'].sum(0)).tolist()
    summary['documents']=n
    return dict(summary=summary,documents=rows if keep_records else [])


@torch.no_grad()
def evaluate_blocks(model,data,scale,documents=32):
    rows=[]
    for i in range(documents):
        q=data['q'][i,:,512:].float();k=data['k'][i,:,:512].float();v=data['v'][i,:,:512].double()
        lf=model.log_feature(q,'q').double();lg=model.log_feature(k,'k').double()
        # Exact positive diagonal reweighting and row scaling; cancelled only in LA evaluation.
        key_scale=lg.amax(1,keepdim=True)
        qcombined=lf+key_scale;query_scale=qcombined.amax(-1,keepdim=True)
        qf=(qcombined-query_scale).exp();kf=(lg-key_scale).exp()
        matrix=qf@kf.transpose(-1,-2)
        denominator=matrix.sum(-1,keepdim=True)
        assert (denominator>0).all()
        ahat=matrix/denominator
        logits=q.double()@k.double().transpose(-1,-2)/math.sqrt(128)
        a=logits.softmax(-1);y=a@v;yh=ahat@v
        pred=matrix*query_scale.exp();target=(logits-scale.double()[:,None,None]).exp()
        rows.append(dict(ordinal=i,attention_nmse=((ahat-a).square().sum((-2,-1))/a.square().sum((-2,-1))).tolist(),
            output_nmse=((yh-y).square().sum((-2,-1))/y.square().sum((-2,-1))).tolist(),
            raw_sse=(pred-target).square().sum((-2,-1)).tolist(),raw_energy=target.square().sum((-2,-1)).tolist(),
            row_l1=(ahat-a).abs().sum(-1).mean(-1).tolist()))
    summary={key:torch.tensor([r[key] for r in rows]).mean(0).tolist() for key in ['attention_nmse','output_nmse','row_l1']}
    summary['raw_nmse']=(torch.tensor([r['raw_sse'] for r in rows],dtype=torch.float64).sum(0)/
                         torch.tensor([r['raw_energy'] for r in rows],dtype=torch.float64).sum(0)).tolist()
    return dict(summary=summary,documents=rows)


def clip_heads(model):
    sums=torch.zeros(4,device='cuda')
    for p in model.parameters():sums+=p.grad.flatten(1).square().sum(-1)
    scale=(10/sums.sqrt().clamp_min(1e-20)).clamp_max(1)
    for p in model.parameters():p.grad.mul_(scale.reshape(4,*([1]*(p.ndim-1))))


def fit(args,data,manifest,method,loss,seed,n,budget):
    name=f'{loss}_{method}_n{n}_b{budget}_s{seed}';path=args.out/(name+'.json')
    if path.exists():print(json.dumps(dict(event='skip',name=name)),flush=True);return
    torch.manual_seed(seed);normalizer,scale=calibration(data['train'],n)
    model=FeaturePair(method,normalizer,budget).cuda()
    params=sum(p.numel() for p in model.parameters())//4
    assert params==2*(36864*budget+64)
    order=torch.randperm(n,generator=torch.Generator().manual_seed(20260906+seed)).tolist()
    assert len(set(order))==n
    digest=hashlib.sha256(json.dumps(order).encode()).hexdigest()
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-4)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,n,eta_min=.0002)
    history=[];start=time.perf_counter()
    for step,index in enumerate(order,1):
        q,k=[data['train'][side][index].float() for side in ['q','k']]
        target=data['train']['logits'][index]-scale[:,None]
        pred=model.log_kernel_pairs(q,k)
        perhead=loss_per_pair(pred,target,loss).mean(-1);value=perhead.mean()
        if not torch.isfinite(value):raise RuntimeError(f'Nonfinite loss: {name} step={step}')
        opt.zero_grad(set_to_none=True);value.backward();clip_heads(model);opt.step();schedule.step()
        if step in [1,64,128,256,512,1024,2048] or step==n:
            val=evaluate_pairs(model,data['validation'],scale,n=32,keep_records=False)['summary']
            row=dict(documents_seen=step,loss=float(value.detach()),validation=val,seconds=time.perf_counter()-start)
            history.append(row);print(json.dumps(dict(event='progress',name=name,**row)),flush=True)
    train_seconds=time.perf_counter()-start
    train=evaluate_pairs(model,data['train'],scale,n=n,keep_records=False)
    val=evaluate_pairs(model,data['validation'],scale)
    test=evaluate_pairs(model,data['test'],scale)
    block=evaluate_blocks(model,data['test']['full'],scale)
    q,k=[data['train'][side][0].float() for side in ['q','k']]
    with torch.no_grad():
        for _ in range(5):model.log_kernel_pairs(q,k)
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
        for _ in range(30):model.log_kernel_pairs(q,k)
        end.record();torch.cuda.synchronize();ms=begin.elapsed_time(end)/30
    result=dict(name=name,method=method,loss=loss,seed=seed,training_documents=n,parameters_per_qk_pair=params,
        budget_scale=budget,m=64,head_labels=manifest['head_labels'],epochs=1,steps=n,unique_pairs_per_head=n*512,
        document_order_sha256=digest,log_scale=scale.tolist(),lr=.002,weight_decay=1e-4,
        selection='final one-epoch checkpoint, no early stopping or test-based choice',
        training_seconds=train_seconds,feature_512_pairs_ms=ms,history=history,
        train=train,validation=val,test=test,test_block=block,
        architecture=('128 -> 192*budget -> 64; SiLU' if method=='mlp' else
          '128 -> 16*budget -> 64; two cubic spline layers' if method=='kan' else
          '128 -> (9*budget addition + 3*budget binary-product nodes) -> (32 addition + 32 binary-product outputs); two cubic spline layers'),
        notes=['One document update, 512 unique legal pairs; every document and Q/K pair appears once per run.',
               'All methods use the same pairs and doc order; exactly matched active parameter counts.',
               'Two trainable layers, no intermediate bias, 64 output biases per branch; softplus features.',
               'Raw-kernel MSE is evaluation only; no row-normalized labels used.',
               'Training data size and optimizer step count both change in the one-pass size sweep.'])
    path.write_text(json.dumps(result,indent=2))
    torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},metadata={k:v for k,v in result.items()
               if k not in ['history','train','validation','test','test_block']}),args.out/(name+'.pt'))
    print(json.dumps(dict(event='complete',name=name,parameters=params,seconds=train_seconds,test=test['summary'],block=block['summary'])),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=P/'fits')
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args();args.out.mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    manifest,data=load_data()
    if args.smoke:
        for method in ['mlp','kan','mulkan']:fit(args,data,manifest,method,'logcosh',11,64,1)
        return
    for n in [64,512,4096]:
        for seed in [11,29,47]:
            for method in ['mlp','kan','mulkan']:fit(args,data,manifest,method,'logcosh',seed,n,1)
    for loss,budget in [('poisson',1),('logcosh',2)]:
        for seed in [11,29,47]:
            for method in ['mlp','kan','mulkan']:fit(args,data,manifest,method,loss,seed,4096,budget)


if __name__=='__main__':main()
