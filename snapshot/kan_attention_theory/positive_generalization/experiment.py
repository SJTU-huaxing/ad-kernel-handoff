"""Train-only positive exponential features; frozen real Qwen Q/K, raw exp kernel."""
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import torch
from torch import nn

P=Path(__file__).resolve().parent
DATA=P.parent/'single_pass_mulkan/data'
LABELS=[[14,0],[14,6],[27,0],[27,6]]
RANKS=[16,32,64,128]
SEEDS=[11,29,47]
D=128

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False))

def digest(x):return hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()

@torch.no_grad()
def load_raw():
    manifest=json.loads((DATA/'manifest.json').read_text());raw={}
    for split in ['train','validation','test']:
        batches=[torch.load(DATA/r['file'],weights_only=True) for r in manifest['shards'] if r['split']==split]
        raw[split]={side:torch.cat([b[side] for b in batches]).permute(1,0,2,3).contiguous().cuda()
                    for side in ['q','k']+(['v'] if split!='train' else [])}
    return manifest,raw

def root(cov,power):
    s,u=torch.linalg.eigh(cov)
    return (u*s.clamp_min(1e-15).pow(power)[:,None])@u.transpose(-1,-2)

@torch.no_grad()
def calibrate(raw):
    cal={};moments={}
    for side in ['q','k']:
        x=raw[side].flatten(1,2);n=x.shape[1]
        mean=torch.zeros(4,D,device='cuda',dtype=torch.float64)
        moment=torch.zeros(4,D,D,device='cuda',dtype=torch.float64)
        for b in range(0,n,8192):
            z=x[:,b:b+8192].double();mean+=z.sum(1);moment+=z.transpose(-1,-2)@z
        mean/=n;moment/=n
        cov=(moment-mean[:,:,None]*mean[:,None,:])/math.sqrt(D)
        cal[side+'_mean']=mean;cal[side+'_cov']=cov;moments[side]=moment/math.sqrt(D)
    identity=torch.eye(D,device='cuda',dtype=torch.float64)[None].expand(4,-1,-1)
    # Ridge chooses a stable invertible coordinate system, without modifying q^T k.
    cq=cal['q_cov']+identity*cal['q_cov'].diagonal(dim1=-2,dim2=-1).mean(-1)[:,None,None]*1e-6
    ck=cal['k_cov']+identity*cal['k_cov'].diagonal(dim1=-2,dim2=-1).mean(-1)[:,None,None]*1e-6
    rq,rk=root(cq,.5),root(ck,.5)
    u,s,vh=torch.linalg.svd(rq@rk)
    tq=root(cq,-.5)@u*s.sqrt()[:,None]
    tk=root(ck,-.5)@vh.transpose(-1,-2)*s.sqrt()[:,None]
    balanced_cov=tq.transpose(-1,-2)@cal['q_cov']@tq+tk.transpose(-1,-2)@cal['k_cov']@tk
    eig,rotation=torch.linalg.eigh(balanced_cov);eig=eig.flip(-1).clamp_min(0);rotation=rotation.flip(-1)
    cal.update(q_transform=tq@rotation,k_transform=tk@rotation,couplings=s,balanced_sum_eigenvalues=eig)
    mq=cal['q_mean']*D**(-.25);mk=cal['k_mean']*D**(-.25)
    es,us=torch.linalg.eigh(moments['q']+moments['k']+mq[:,:,None]*mk[:,None,:]+mk[:,:,None]*mq[:,None,:])
    cal.update(raw_sderf_eigenvalues=es.flip(-1).clamp_min(0),raw_sderf_rotation=us.flip(-1))
    cal['identity']=identity
    error=(cal['q_transform']@cal['k_transform'].transpose(-1,-2)-identity).abs().amax().item()
    assert error<1e-7,error
    save(P/'results/calibration.json',dict(head_labels=LABELS,samples_per_marginal=raw['q'].shape[1]*512,
        transform_inverse_error=error,couplings=s.tolist(),balanced_sum_eigenvalues=eig.tolist(),
        q_mean_norm=cal['q_mean'].norm(dim=-1).tolist(),k_mean_norm=cal['k_mean'].norm(dim=-1).tolist(),
        raw_cov_trace={side:cal[side+'_cov'].diagonal(dim1=-2,dim2=-1).sum(-1).tolist() for side in ['q','k']}))
    torch.save({k:v.cpu() for k,v in cal.items()},P/'results/calibration.pt')
    return cal

def a_opt(lam):return (1-2*lam-torch.sqrt((2*lam+1).square()+8*lam))/16

def nodes(seed,kind,m=128,heads=4):
    if kind=='sobol':
        vals=[]
        for h in range(heads):
            u=torch.quasirandom.SobolEngine(D,scramble=True,seed=seed+100*h).draw(m,dtype=torch.float64)
            vals.append(math.sqrt(2)*torch.erfinv(2*u-1))
        return torch.stack(vals).cuda()
    gen=torch.Generator(device='cuda').manual_seed(seed)
    g=torch.randn(heads,D,D,device='cuda',dtype=torch.float64,generator=gen)
    q,r=torch.linalg.qr(g);q*=r.diagonal(dim1=-2,dim2=-1).sign()[:,None]
    radius=torch.randn(heads,D,D,device='cuda',dtype=torch.float64,generator=gen).norm(dim=-1)
    return (q.transpose(-1,-2)*radius[:,:,None])[:,:m]

class Features(nn.Module):
    def __init__(self,cal,m,seed,method='balanced_sderf_sobol',learn=None):
        super().__init__();self.m=m;self.method=method;self.learn=learn
        balanced=method.startswith('balanced')
        for side in ['q','k']:
            self.register_buffer(side+'_mean',cal[side+'_mean'].clone() if balanced else torch.zeros_like(cal[side+'_mean']))
            transform=cal[side+'_transform'] if balanced else cal['raw_sderf_rotation'] if method.startswith('sderf') else cal['identity']
            self.register_buffer(side+'_transform',transform.clone())
        omega=nodes(seed,'sobol' if 'sobol' in method else 'orf',m)
        if 'sderf' in method:
            aa=a_opt(cal['balanced_sum_eigenvalues'] if balanced else cal['raw_sderf_eigenvalues'])
            w=2*(omega.square()*aa[:,None]).sum(-1)+.5*torch.log1p(-4*aa).sum(-1)[:,None]-math.log(m)
            omega=omega*torch.sqrt(1-4*aa)[:,None]
        else:w=torch.full((4,m),-math.log(m),device='cuda',dtype=torch.float64)
        if learn:
            self.omega=nn.Parameter(omega.float());self.log_weight=nn.Parameter(w.float())
            if learn=='untied':self.omega_k=nn.Parameter(omega.float().clone())
        else:self.register_buffer('omega',omega);self.register_buffer('log_weight',w)

    def transform(self,x,side):
        mean=getattr(self,side+'_mean');other=getattr(self,('k' if side=='q' else 'q')+'_mean')
        xc=x-mean[:,None]
        z=(xc*D**(-.25))@getattr(self,side+'_transform')
        correction=(xc*other[:,None]).sum(-1)/math.sqrt(D)+(mean*other).sum(-1)[:,None]/(2*math.sqrt(D))
        base=correction-z.square().sum(-1)/2
        return z,base

    def prepared(self,z,base,side):
        omega=self.omega_k if side=='k' and self.learn=='untied' else self.omega
        return z@omega.transpose(-1,-2)+base[:,:,None]+self.log_weight[:,None]/2

    def log_feature(self,x,side):return self.prepared(*self.transform(x,side),side)

    def paired(self,zq,zk,bq,bk):
        return (self.prepared(zq,bq,'q')+self.prepared(zk,bk,'k')).logsumexp(-1)

@torch.no_grad()
def prepare(raw,cal):
    f=Features(cal,16,11);data={};audit={}
    for split,rr in raw.items():
        q=rr['q'] if split=='train' else rr['q'][:,:,512:]
        k=rr['k'] if split=='train' else rr['k'][:,:,:512]
        q=q.flatten(1,2);k=k.flatten(1,2);n=q.shape[1]
        perm=torch.randperm(n,generator=torch.Generator().manual_seed({'train':20260908,'validation':20270908,'test':20280908}[split]))
        assert len(torch.unique(perm))==n
        rec={side:torch.empty(4,n,D,device='cuda',dtype=torch.float32) for side in ['zq','zk']}
        rec.update({key:torch.empty(4,n,device='cuda',dtype=torch.float32) for key in ['bq','bk','target']})
        max_error=0.
        for b in range(0,n,8192):
            x=q[:,b:b+8192].double();y=k[:,perm[b:b+8192].cuda()].double()
            zq,bq=f.transform(x,'q');zk,bk=f.transform(y,'k')
            target=(x*y).sum(-1)/math.sqrt(D)
            restored=(zq*zk).sum(-1)+bq+bk+(zq.square().sum(-1)+zk.square().sum(-1))/2
            max_error=max(max_error,(target-restored).abs().max().item())
            for key,value in dict(zq=zq,zk=zk,bq=bq,bk=bk,target=target).items():rec[key][:,b:b+8192]=value.float()
        data[split]=rec;audit[split]=dict(pairs=n,key_permutation_sha256=digest(perm),unique_key_indices=n,exact_dot_identity_max_error=max_error)
        print(json.dumps(dict(event='prepared',split=split,**audit[split])),flush=True)
    # Common head scale for numerical conditioning; restored in every raw-kernel prediction.
    scale=data['train']['target'].double().logsumexp(-1)-math.log(data['train']['target'].shape[1])
    save(P/'results/data_audit.json',dict(splits=audit,scale=scale.tolist(),pair_measure='Globally permuted independent empirical Q/K marginals; each Q and K vector occurs once per split.'))
    return data,scale.float()

@torch.no_grad()
def paired_metrics(model,rec,scale,n=None):
    n=n or rec['target'].shape[1];sums=torch.zeros(6,4,device='cuda',dtype=torch.float64)
    for b in range(0,n,4096):
        lp=model.paired(*[rec[x][:,b:b+4096] for x in ['zq','zk','bq','bk']]).double()
        lt=rec['target'][:,b:b+4096].double();r=lp-lt
        y=(lt-scale.double()[:,None]).exp();yh=(lp-scale.double()[:,None]).exp()
        sums+=torch.stack([(yh-y).square().sum(-1),y.square().sum(-1),(yh-y-y*r).sum(-1),y.sum(-1),
             (r.abs()+torch.nn.functional.softplus(-2*r.abs())-math.log(2)).sum(-1),(r.abs()<math.log(2)).double().sum(-1)])
    return dict(raw_nmse=(sums[0]/sums[1]).tolist(),idiv_per_mass=(sums[2]/sums[3]).tolist(),logcosh=(sums[4]/n).tolist(),factor2=(sums[5]/n).tolist())

@torch.no_grad()
def match_mass(model,rec,n=65536):
    lp=[];lt=[]
    for b in range(0,min(n,rec['target'].shape[1]),4096):
        lp.append(model.paired(*[rec[x][:,b:b+4096] for x in ['zq','zk','bq','bk']]).double())
        lt.append(rec['target'][:,b:b+4096].double())
    shift=torch.cat(lt,1).logsumexp(-1)-torch.cat(lp,1).logsumexp(-1)
    model.log_weight.add_(shift[:,None].to(model.log_weight.dtype))
    return shift.tolist()

def fit(data,cal,scale,m,seed,kind,out,smoke=False,loss='idiv'):
    name=f'{kind}_m{m}_s{seed}';path=out/(name+'.json')
    if path.exists():return
    model=Features(cal,m,seed,learn=kind).cuda()
    shift=match_mass(model,data['train']);params=sum(p.numel() for p in model.parameters())//4
    n=data['train']['target'].shape[1] if not smoke else 32768
    batch=512;steps=n//batch
    # Fixed global permutation of Q/K pairs; no pair is revisited for gradients.
    order=torch.randperm(n,generator=torch.Generator().manual_seed(seed+20260908)).cuda()
    opt=torch.optim.Adam(model.parameters(),lr=.003)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=.0003)
    history=[];start=time.perf_counter()
    history.append(dict(step=0,validation=paired_metrics(model,data['validation'],scale,n=16384)))
    for step in range(1,steps+1):
        ix=order[(step-1)*batch:step*batch]
        pred=model.paired(*[data['train'][x][:,ix] for x in ['zq','zk','bq','bk']])-scale[:,None]
        target=data['train']['target'][:,ix]-scale[:,None]
        if loss=='idiv':value=(pred.exp()-target.exp()*pred).mean()
        elif loss=='beta15':
            beta=1.5
            # Target-only constant omitted. D_beta(y || yhat), beta strictly below 2.
            value=(((beta-1)*(beta*pred).exp()-beta*(target+(beta-1)*pred).exp())/(beta*(beta-1))).mean()
        else:raise ValueError(loss)
        if not torch.isfinite(value):raise RuntimeError((name,step,float(value)))
        opt.zero_grad(set_to_none=True);value.backward()
        norm=sum(p.grad.flatten(1).square().sum(-1) for p in model.parameters()).sqrt()
        factor=(10/norm.clamp_min(1e-20)).clamp_max(1)
        for p in model.parameters():p.grad.mul_(factor.reshape(4,*([1]*(p.ndim-1))))
        opt.step();schedule.step()
        if step in [256,512,1024,2048,3072,3584,3840,4096] or step==steps:
            row=dict(step=step,unique_pairs_seen=step*batch,validation=paired_metrics(model,data['validation'],scale,n=16384),seconds=time.perf_counter()-start)
            history.append(row);print(json.dumps(dict(event='training',name=name,**row)),flush=True)
    result=dict(name=name,kind=kind,m=m,seed=seed,loss=loss,loss_note='Raw kernel Bregman divergence, head-constant scaling for numerical conditioning',
        parameters_per_head=params,epochs=1,steps=steps,unique_pairs_per_head=n,pair_order_sha256=digest(order),initial_mass_shift=shift,
        selection='Final single-pass checkpoint; fixed hyperparameters, no validation/test checkpoint or hyperparameter selection',
        history=history,train=paired_metrics(model,data['train'],scale),validation=paired_metrics(model,data['validation'],scale),
        test=paired_metrics(model,data['test'],scale),seconds=time.perf_counter()-start)
    save(path,result)
    torch.save(dict(state={k:v.cpu() for k,v in model.state_dict().items()},metadata={k:v for k,v in result.items() if k not in ['history','train','validation','test']}),out/(name+'.pt'))
    print(json.dumps(dict(event='fit_complete',name=name,test=result['test'],seconds=result['seconds'])),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--loss',default='idiv',choices=['idiv','beta15']);args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    out=P/('smoke' if args.smoke else 'fits' if args.loss=='idiv' else 'beta15_fits');out.mkdir(exist_ok=True)
    manifest,raw=load_raw();cal=calibrate(raw['train']);data,scale=prepare(raw,cal)
    del raw
    hashes={s:{d['token_sha256'] for d in manifest['documents'] if d['split']==s} for s in ['train','validation','test']}
    assert all(not hashes[a]&hashes[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
    save(P/('results/protocol.json' if args.loss=='idiv' else 'results/beta15_protocol.json'),dict(model=manifest['model'],revision=manifest['model_revision'],data_revision=manifest['data_revision'],
        labels=LABELS,ranks=RANKS,seeds=SEEDS,documents={s:len(v) for s,v in hashes.items()},
        data_note='WikiText103 source train split partitioned by document into 4096/128/256; not official benchmark test.',
        training_measure='Empirical product marginals via one global K permutation. Every Q and K vector used once per run.',
        objective=f'Unnormalized exp(q^T k / sqrt(128)); {args.loss}, no MSE training.',
        study_status='Original planned experiment' if args.loss=='idiv' else 'Exploratory loss ablation prompted by original holdout overshoot results; reused test sets, not a fresh confirmatory test.',
        predeclared_gate='At m=64, median-seed learned error <=75% of each train-calibrated RF baseline on >=3/4 heads, and <=10 times achieved transductive NMF error on >=3/4 heads. Apply on both old official-test diagnostic and new internal holdout.',
        transductive_warning='NMF sees every tested matrix entry. Learned feature maps and RF calibration use training documents only.'))
    for seed in ([11] if args.smoke else SEEDS):
        for m in ([64] if args.smoke or args.loss=='beta15' else RANKS):
            for kind in ['shared','untied']:fit(data,cal,scale,m,seed,kind,out,args.smoke,args.loss)
    save(out/'completion.json',dict(fits=len(list(out.glob('*_m*_s*.json')))))

if __name__=='__main__':main()
