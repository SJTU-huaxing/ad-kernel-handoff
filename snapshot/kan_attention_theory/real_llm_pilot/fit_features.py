"""Train independent per-head feature maps on frozen, document-split LLM Q/K."""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class HeadLinear(nn.Module):
    def __init__(self, heads, inputs, outputs):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(heads, outputs, inputs) / math.sqrt(inputs))
        self.bias = nn.Parameter(torch.zeros(heads, outputs))

    def forward(self, x):
        return torch.bmm(x, self.weight.transpose(1, 2)) + self.bias[:, None]


class HeadSpline(nn.Module):
    """Cubic B-spline KAN edge functions plus the usual SiLU base functions."""
    def __init__(self, heads, inputs, outputs, grid=8, degree=3):
        super().__init__()
        self.inputs, self.grid, self.degree = inputs, grid, degree
        self.base = HeadLinear(heads, inputs, outputs)
        self.coefficients = nn.Parameter(
            torch.randn(heads, outputs, inputs * (grid + degree)) * .03 / math.sqrt(inputs)
        )
        self.register_buffer('knots', torch.arange(-degree, grid + degree + 1).float() * (8 / grid) - 4)

    def basis(self, x):
        x = x.unsqueeze(-1)
        knots = self.knots
        bases = ((x >= knots[:-1]) & (x < knots[1:])).to(x.dtype)
        for order in range(1, self.degree + 1):
            bases = ((x - knots[:-(order+1)]) * bases[..., :-1]
                     + (knots[order+1:] - x) * bases[..., 1:]) / (order * 8 / self.grid)
        return bases

    def forward(self, x):
        bases = self.basis(x).flatten(-2)
        return self.base(F.silu(x)) + torch.bmm(bases, self.coefficients.transpose(1, 2))


class FeaturePair(nn.Module):
    def __init__(self, heads, d, m, method, normalization, amplitude, grid=8, width=16):
        super().__init__()
        self.method, self.m, self.heads = method, m, heads
        for key, value in normalization.items():
            self.register_buffer(key, value)
        self.register_buffer('amplitude', amplitude[:, None, None])
        if method.startswith('kan'):
            build = lambda: nn.Sequential(HeadSpline(heads,d,width,grid),HeadSpline(heads,width,m,grid))
            self.qnet, self.knet = build(), build()
            self.hidden_width = width
        elif method.startswith('mlp'):
            kan_per_branch = (d*width + width*m)*(grid+4) + width+m
            hidden = max(1, round((kan_per_branch-m)/(d+m+1)))
            build = lambda: nn.Sequential(HeadLinear(heads,d,hidden),nn.SiLU(),HeadLinear(heads,hidden,m))
            self.qnet, self.knet = build(), build()
            self.hidden_width = hidden
        elif method == 'hedgehog':
            if m % 2:
                raise ValueError('Hedgehog feature dimension must be even')
            self.qnet, self.knet = HeadLinear(heads,d,m//2), HeadLinear(heads,d,m//2)
            self.hidden_width = m//2
        else:
            raise ValueError(method)

    def feature(self, x, side):
        x = (x - getattr(self,side+'_mean')[:,None]) / getattr(self,side+'_std')[:,None]
        z = getattr(self,side+'net')(x)
        if self.method == 'hedgehog':
            z = torch.cat([z.softmax(-1),(-z).softmax(-1)],-1) * (self.m/2)
        elif self.method.endswith('positive'):
            z = F.softplus(z) / math.log(2)
        else:
            z = 1 + z
        return z * self.amplitude / math.sqrt(self.m)


def read_data(root, manifest, indices):
    result={}
    for split in ['train','validation','test','test_long','ood']:
        records=[]
        for doc in manifest['documents']:
            if doc['split'] != split:
                continue
            data=torch.load(root/doc['file'],weights_only=True,map_location='cpu')
            records.append(dict(file=doc['file'],q=data['q'][indices].cuda(),
                                k=data['k'][indices].cuda(),v=data['v'][indices].cuda()))
        result[split]=records
    return result


def target_for(q,k,positions,objective,log_scale):
    logits=q@k.transpose(-1,-2)/math.sqrt(q.shape[-1])
    mask=(torch.arange(k.shape[1],device=q.device)[None,:]<=positions[:,None] if positions.ndim==1 else
          torch.arange(k.shape[1],device=q.device)[None,None,:]<=positions[:,:,None])
    logits=logits.masked_fill(~mask,-torch.inf)
    attention=logits.softmax(-1)
    if objective=='raw':
        target=(logits-log_scale[:,None,None]).exp()
    else:
        target=attention*mask.sum(-1)[None,:,None]
    return target,attention,mask


@torch.no_grad()
def calibrate_raw(records):
    sums=[];sums2=[];counts=[]
    for record in records:
        q,k=record['q'].float(),record['k'].float()
        pos=torch.linspace(128,q.shape[1]-1,32,device='cuda').long()
        logits=(q[:,pos].double()@k.double().transpose(-1,-2))/math.sqrt(q.shape[-1])
        mask=torch.arange(k.shape[1],device='cuda')[None,:]<=pos[:,None]
        logits=logits.masked_fill(~mask,-torch.inf)
        sums.append(logits.flatten(1).logsumexp(-1))
        sums2.append((2*logits).flatten(1).logsumexp(-1))
        counts.append(int(mask.sum()))
    log_mean=torch.stack(sums).logsumexp(0)-math.log(sum(counts))
    log_mean2=torch.stack(sums2).logsumexp(0)-math.log(sum(counts))
    scale=.5*log_mean2
    amplitude=((log_mean-scale)/2).exp().float()
    return scale.float(),amplitude


@torch.no_grad()
def evaluate(model, records, objective, log_scale, block=False, count=None, exact=False):
    results=[]
    for record in records[:count]:
        dtype=next(model.parameters()).dtype if exact else torch.float32
        q,k,v=record['q'].to(dtype),record['k'].to(dtype),record['v'].to(dtype)
        positions=torch.arange(q.shape[1]-512,q.shape[1],device='cuda')
        q=q[:,positions]
        if block:
            k,v=k[:,:512],v[:,:512]
        target,a,mask=target_for(q,k,positions,objective,log_scale)
        pred=model.feature(q,'q')@model.feature(k,'k').transpose(-1,-2)
        pred=pred.masked_fill(~mask,0)
        denominator=pred.sum(-1,keepdim=True)
        if exact:
            if (denominator==0).any():raise RuntimeError('Exactly zero denominator in exact feature evaluation')
            safe_denominator=denominator
        else:
            safe_denominator=torch.where(denominator.abs()<1e-8,torch.full_like(denominator,1e-8),denominator)
        ahat=pred/safe_denominator
        y=a@v;yhat=ahat@v
        row=dict(file=record['file'],
                 kernel_nmse=((pred-target).square().sum((-2,-1))/target.square().sum((-2,-1)).clamp_min(1e-30)).cpu().tolist(),
                 kernel_squared_error=(pred-target).double().square().sum((-2,-1)).cpu().tolist(),
                 kernel_target_energy=target.double().square().sum((-2,-1)).cpu().tolist(),
                 legal_pair_count=int(mask.sum()),
                 attention_nmse=((ahat-a).square().sum((-2,-1))/a.square().sum((-2,-1))).cpu().tolist(),
                 output_nmse=((yhat-y).square().sum((-2,-1))/y.square().sum((-2,-1)).clamp_min(1e-30)).cpu().tolist(),
                 row_l1=(ahat-a).abs().sum(-1).mean(-1).cpu().tolist(),
                 negative_fraction=((pred<0)&mask).sum((-2,-1)).div(mask.sum()).cpu().tolist(),
                 nonpositive_denominator_fraction=(denominator<=0).float().mean((-2,-1)).cpu().tolist(),
                 top1_agreement=(ahat.argmax(-1)==a.argmax(-1)).float().mean(-1).cpu().tolist())
        results.append(row)
    return results


def aggregate(records,key):
    return torch.tensor([x[key] for x in records]).mean(0)


def pooled_kernel_nmse(records):
    error=torch.tensor([r['kernel_squared_error'] for r in records],dtype=torch.float64).sum(0)
    energy=torch.tensor([r['kernel_target_energy'] for r in records],dtype=torch.float64).sum(0)
    return (error/energy.clamp_min(1e-300)).float()


def clip_head_gradients(model, heads, maximum=10):
    norms=torch.zeros(heads,device='cuda')
    for p in model.parameters():
        if p.grad is not None:
            norms+=p.grad.flatten(1).square().sum(-1)
    scales=(maximum/norms.sqrt().clamp_min(1e-12)).clamp_max(1)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(scales.view(heads,*([1]*(p.ndim-1))))


def fit_one(args,data,normalization,labels,method,objective,seed,m):
    name=f'{objective}_{method}_m{m}_s{seed}'
    result_path=args.out/(name+'.json')
    if result_path.exists():
        print(json.dumps(dict(event='skip',name=name)),flush=True)
        return
    torch.manual_seed(seed)
    h=len(labels);d=data['train'][0]['q'].shape[-1]
    sampler=None
    if args.sampling=='energy':
        if objective!='raw':raise ValueError('Importance sampler is defined only for the raw exponential target')
        from raw_importance import RawEnergySampler
        sampler=RawEnergySampler(data['train'])
        log_scale,amplitude=sampler.log_scale,sampler.amplitude
        print(json.dumps(dict(event='sampler_verification',**sampler.verification())),flush=True)
    elif objective=='raw':
        log_scale,amplitude=calibrate_raw(data['train'])
    else:
        log_scale=torch.zeros(h,device='cuda');amplitude=torch.ones(h,device='cuda')
    model=FeaturePair(h,d,m,method,normalization,amplitude,args.grid,args.width).cuda()
    parameters=sum(p.numel() for p in model.parameters())//h
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.steps,eta_min=args.lr*.1)
    best=torch.full((h,),torch.inf);best_steps=torch.zeros(h,dtype=torch.int64)
    best_state=copy.deepcopy(model.state_dict())
    history=[];start=time.perf_counter()
    generator=torch.Generator(device='cuda').manual_seed(9000+seed)
    for step in range(1,args.steps+1):
        if sampler:
            q,k,positions,importance_weight=sampler.sample(generator)
        else:
            index=int(torch.randint(len(data['train']),(1,),generator=generator,device='cuda'))
            record=data['train'][index]
            length=record['q'].shape[1]
            positions=torch.randint(128,length,(32,),generator=generator,device='cuda')
            q=record['q'][:,positions].float();k=record['k'].float()
        target,_,mask=target_for(q,k,positions,objective,log_scale)
        pred=model.feature(q,'q')@model.feature(k,'k').transpose(-1,-2)
        error=(pred-target).square()*mask
        if sampler:
            per_head=(error.sum(-1)*importance_weight).mean(-1)
        elif objective=='raw':
            per_head=error.sum((-2,-1))/mask.sum()
        else:
            per_head=error.sum((-2,-1))/target.square().sum((-2,-1)).clamp_min(1e-20)
        loss=per_head.mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite training loss: {name}, step={step}, per_head={per_head}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_head_gradients(model,h)
        optimizer.step();schedule.step()
        if step==1 or step%200==0 or step==args.steps:
            model.eval()
            val=evaluate(model,data['validation'],objective,log_scale,count=6)
            score=pooled_kernel_nmse(val) if objective=='raw' else aggregate(val,'kernel_nmse')
            improved=score<best
            for key,value in model.state_dict().items():
                if value.ndim and value.shape[0]==h and 'knots' not in key:
                    best_state[key][improved.to(value.device)]=value[improved.to(value.device)]
            best[improved]=score[improved];best_steps[improved]=step
            row=dict(step=step,train_loss=float(loss),validation_kernel_nmse=score.tolist(),
                     seconds=time.perf_counter()-start)
            history.append(row)
            print(json.dumps(dict(event='training',name=name,**row)),flush=True)
            model.train()
    model.load_state_dict(best_state);model.eval()
    training_seconds=time.perf_counter()-start
    evaluations={}
    for split in ['validation','test','test_long','ood']:
        evaluations[split]=evaluate(model,data[split],objective,log_scale)
    evaluations['test_block']=evaluate(model,data['test'],objective,log_scale,block=True)
    q=data['test'][0]['q'][:,:512].float();k=data['test'][0]['k'][:,:512].float()
    with torch.no_grad():
        for _ in range(5):model.feature(q,'q');model.feature(k,'k')
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(30):model.feature(q,'q');model.feature(k,'k')
        end.record();torch.cuda.synchronize()
        feature_ms=begin.elapsed_time(end)/30
    result=dict(name=name,method=method,objective=objective,seed=seed,m=m,head_labels=labels,
                parameters_per_head_pair=parameters,hidden_width=model.hidden_width,
                grid=args.grid,steps=args.steps,lr=args.lr,weight_decay=1e-4,
                sampling=args.sampling,sampler_verification=sampler.verification() if sampler else None,
                log_scale=log_scale.cpu().tolist(),best_validation_steps=best_steps.tolist(),
                checkpoint_selection=('pooled raw-kernel squared error / pooled target energy'
                                      if objective=='raw' else 'mean document kernel NMSE'),
                training_seconds=training_seconds,feature_pair_512tokens_ms=feature_ms,
                history=history,evaluations=evaluations,
                objective_definition=('E[(exp(logit-c_h)-f(q)^T g(k))^2], c_h from train RMS kernel'
                                      if objective=='raw' else
                                      'Relative squared error fitting n_visible*softmax(logits) with separable f(q)^T g(k)'),
                warnings=['All head maps are independent; model weights are frozen.',
                          'Validation chooses checkpoints per head; test data never enter optimization.',
                          'Kernel NMSE objectives differ; compare attention/output NMSE across objectives.',
                          'Feature timing is eager PyTorch, not a fused production attention benchmark.'])
    result_path.write_text(json.dumps(result,indent=2))
    torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},
                    metadata={k:v for k,v in result.items() if k not in ['history','evaluations']}),args.out/(name+'.pt'))
    print(json.dumps(dict(event='complete',name=name,parameters=parameters,seconds=training_seconds,
                          test_attention_nmse=aggregate(evaluations['test'],'attention_nmse').tolist(),
                          test_output_nmse=aggregate(evaluations['test'],'output_nmse').tolist())),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--objective',choices=['raw','row'],default='row')
    p.add_argument('--methods',nargs='+',default=['mlp_signed','kan_signed','mlp_positive','kan_positive','hedgehog'])
    p.add_argument('--seeds',nargs='+',type=int,default=[11,29])
    p.add_argument('--m',nargs='+',type=int,default=[64])
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--lr',type=float,default=.002)
    p.add_argument('--grid',type=int,default=8)
    p.add_argument('--width',type=int,default=16)
    p.add_argument('--sampling',choices=['uniform','energy'],default='uniform')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    manifest=json.loads((args.data/'manifest.json').read_text())
    labels=[(l,h) for l in ([14,27] if args.objective=='raw' else [0,14,27]) for h in [0,6]]
    indices=[manifest['head_labels'].index(list(label)) for label in labels]
    norm=torch.load(args.analysis/'normalization.pt',weights_only=True)
    normalization={key:norm[key][indices].cuda() for key in ['q_mean','q_std','k_mean','k_std']}
    data=read_data(args.data,manifest,indices)
    print(json.dumps(dict(event='loaded',objective=args.objective,labels=labels,
                          documents={k:len(v) for k,v in data.items()},gpu=torch.cuda.get_device_name())),flush=True)
    for m in args.m:
        for seed in args.seeds:
            for method in args.methods:
                fit_one(args,data,normalization,labels,method,args.objective,seed,m)


if __name__=='__main__':
    main()
