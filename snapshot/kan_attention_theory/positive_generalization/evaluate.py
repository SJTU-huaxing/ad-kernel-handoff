"""Float64 raw-kernel holdout evaluation; test-matrix NMF is only a reference."""
import argparse, math, sys, time
from pathlib import Path
import torch
from experiment import P, DATA, LABELS, RANKS, SEEDS, Features, a_opt, root, save, json

sys.path.insert(0,str(P.parent/'gaussian_go_nogo'))
from run import decompose, nmf_curve, log_sse

@torch.no_grad()
def load_eval():
    manifest=json.loads((DATA/'manifest.json').read_text());pools={}
    shards=[torch.load(DATA/r['file'],weights_only=True) for r in manifest['shards'] if r['split']=='test']
    pools['internal']={side:torch.cat([b[side] for b in shards]).permute(1,0,2,3).contiguous().cuda() for side in ['q','k','v']}
    old=P.parent/'real_llm_pilot/data_qwen25_1p5b'
    om=json.loads((old/'manifest.json').read_text());hi=[om['head_labels'].index(h) for h in LABELS]
    docs=[d for d in om['documents'] if d['split']=='test'];boxes=[torch.load(old/d['file'],weights_only=True) for d in docs]
    pools['official']={side:torch.stack([b[side][hi] for b in boxes],1).cuda() for side in ['q','k','v']}
    pools['official']['docs']=docs
    # Official old evaluation documents were excluded when the large corpus was assembled.
    train_hashes={d['token_sha256'] for d in manifest['documents'] if d['split']=='train'}
    official_hashes={d['token_sha256'] for d in docs}
    assert not train_hashes&official_hashes
    train=[]
    for rr in [r for r in manifest['shards'] if r['split']=='train'][:2]:train.append(torch.load(DATA/rr['file'],weights_only=True))
    tq=torch.cat([b['q'] for b in train]).permute(1,0,2,3).flatten(1,2).cuda().double()
    tk=torch.cat([b['k'] for b in train]).permute(1,0,2,3).flatten(1,2).cuda().double()
    perm=torch.randperm(4096*512,generator=torch.Generator().manual_seed(20260908))[:65536]
    # Reconstruct precisely the initialization's training-only product pairs across all shards.
    all_k=[]
    for rr in [r for r in manifest['shards'] if r['split']=='train']:
        bb=torch.load(DATA/rr['file'],weights_only=True);all_k.append(bb['k'])
    all_k=torch.cat(all_k).permute(1,0,2,3).flatten(1,2)
    tk=all_k[:,perm].cuda().double()
    return pools,(tq,tk)

@torch.no_grad()
def extra_calibration(cal):
    # ADERF, Theorem 4.2, using uncentered training second moments.
    mq=cal['q_mean']*128**(-.25);mk=cal['k_mean']*128**(-.25)
    cq=cal['q_cov']+mq[:,:,None]*mq[:,None,:]
    ck=cal['k_cov']+mk[:,:,None]*mk[:,None,:]
    rq,rk=root(cq,.5),root(ck,.5);u,s,vh=torch.linalg.svd(rq@rk)
    tq=root(cq,-.5)@u*s.sqrt()[:,None]
    tk=root(ck,-.5)@vh.transpose(-1,-2)*s.sqrt()[:,None]
    phi=(2*s.sum(-1)+2*(mq*mk).sum(-1))/128
    assert (phi>=0).all()
    return tq,tk,a_opt(phi)

def make_baseline(cal,m,seed,method,aderf):
    if method=='aderf_orf':
        f=Features(cal,m,seed,'favor_orf');tq,tk,aa=aderf
        f.q_transform.copy_(tq);f.k_transform.copy_(tk)
        omega=f.omega.clone();f.log_weight.copy_(2*omega.square().sum(-1)*aa[:,None]+64*torch.log1p(-4*aa)[:,None]-math.log(m))
        f.omega.mul_(torch.sqrt(1-4*aa)[:,None,None]);return f
    return Features(cal,m,seed,method.replace('_mass',''))

@torch.no_grad()
def raw_mass_match(f,train):
    preds=[];targets=[]
    for b in range(0,train[0].shape[1],4096):
        q,k=[x[:,b:b+4096] for x in train]
        preds.append((f.log_feature(q,'q')+f.log_feature(k,'k')).logsumexp(-1))
        targets.append((q*k).sum(-1)/math.sqrt(128))
    shift=torch.cat(targets,1).logsumexp(-1)-torch.cat(preds,1).logsumexp(-1)
    f.log_weight.add_(shift[:,None]);return shift.tolist()

@torch.no_grad()
def evaluate_matrix(f,q,k,v=None):
    lq=f.log_feature(q.double(),'q');lk=f.log_feature(k.double(),'k')
    sq=lq.amax((1,2));sk=lk.amax((1,2))
    mat=(lq-sq[:,None,None]).exp()@(lk-sk[:,None,None]).exp().transpose(-1,-2)
    lp=mat.log()+sq[:,None,None]+sk[:,None,None]
    if not torch.isfinite(lp).all():
        lp=torch.cat([(lq[:,b:b+32,None]+lk[:,None]).logsumexp(-1) for b in range(0,len(q[0]),32)],1)
    assert torch.isfinite(lp).all()
    target=q.double()@k.double().transpose(-1,-2)/math.sqrt(128)
    energy=(2*target).flatten(1).logsumexp(-1)
    nmse=log_sse(lp,target)-energy
    maximum=target.amax((1,2));y=(target-maximum[:,None,None]).exp();yh=(lp-maximum[:,None,None]).exp();r=lp-target
    mass=y.sum((1,2));idiv=(yh-y-y*r).sum((1,2))/mass
    result=dict(log_nmse=nmse.tolist(),log_energy=energy.tolist(),idiv_per_mass=idiv.tolist(),
        log_mass=mass.log().add(maximum).tolist(),logcosh=(r.abs()+torch.nn.functional.softplus(-2*r.abs())-math.log(2)).mean((1,2)).tolist(),
        factor2=(r.abs()<math.log(2)).double().mean((1,2)).tolist())
    if v is not None:
        a=target.softmax(-1);ah=lp.softmax(-1);out=a@v.double();outh=ah@v.double()
        result.update(output_nmse=((out-outh).square().sum((1,2))/out.square().sum((1,2))).tolist(),
                      row_l1=(a-ah).abs().sum(-1).mean(-1).tolist())
    return result

def models(cal,train):
    bank=[];aderf=extra_calibration(cal)
    methods=['favor_orf','sderf_orf','aderf_orf','balanced_orf','balanced_sderf_orf','balanced_sderf_sobol','balanced_sderf_sobol_mass']
    for method in methods:
        for seed in [1009+37*i for i in range(10)]:
            for m in RANKS:
                f=make_baseline(cal,m,seed,method,aderf)
                shift=raw_mass_match(f,train) if method.endswith('_mass') else None
                bank.append((dict(method=method,seed=seed,m=m,initial_mass_shift=shift),f))
    for seed in SEEDS:
        for m in RANKS:
            for kind in ['shared','untied']:
                ckpt=torch.load(P/f'fits/{kind}_m{m}_s{seed}.pt',weights_only=True)
                f=Features(cal,m,seed,learn=kind);f.load_state_dict(ckpt['state']);f.double()
                bank.append((dict(method='learned_'+kind,seed=seed,m=m),f))
    save(P/'results/evaluation_models.json',[meta for meta,_ in bank])
    return bank

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--nmf-steps',type=int,default=1600);args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    start=time.perf_counter();pools,train=load_eval()
    cal={k:v.cuda() for k,v in torch.load(P/'results/calibration.pt',weights_only=True).items()}
    bank=models(cal,train);del train
    labels16=[[l,h] for l in [0,7,14,27] for h in [0,3,6,9]];oldhi=[labels16.index(h) for h in LABELS]
    violations=0;checks=0
    for dataset,pool in pools.items():
        qp=pool['q'][:,:,512:].flatten(1,2);kp=pool['k'][:,:,:512].flatten(1,2)
        for source in ['product','same_context']:
            for rep in range(4):
                path=P/f'results/{dataset}_{source}_{rep}.json'
                if path.exists():continue
                if dataset=='official':old=json.loads((P.parent/f'gaussian_go_nogo/results/{source}_{rep}.json').read_text())
                if source=='product':
                    if dataset=='official':qi=torch.tensor(old['sampling']['q_indices']);ki=torch.tensor(old['sampling']['k_indices'])
                    else:
                        gen=torch.Generator().manual_seed(20290908+rep)
                        qi=torch.randperm(qp.shape[1],generator=gen)[:512];ki=torch.randperm(kp.shape[1],generator=gen)[:512]
                    q=qp[:,qi];k=kp[:,ki];v=None;sampling=dict(q_indices=qi.tolist(),k_indices=ki.tolist())
                else:q=pool['q'][:,rep,512:];k=pool['k'][:,rep,:512];v=pool['v'][:,rep,:512];sampling=dict(document_ordinal=rep)
                if dataset=='official':
                    floors={m:[v[h] for h in oldhi] for m,v in old['svd'].items()}
                    nmf=[dict(m=row['m'],error=[row['error'][h] for h in oldhi]) for row in old['nmf']]
                else:
                    A,svd,floors,energy,logits=decompose(q,k);nmf=nmf_curve(A,svd,args.nmf_steps)
                    torch.save(dict(singular_values=svd[1].cpu()),P/f'results/{dataset}_{source}_{rep}_spectrum.pt')
                rows=[]
                for meta,f in bank:
                    row=dict(**meta,**evaluate_matrix(f,q,k,v));rows.append(row)
                    for logerr,floor in zip(row['log_nmse'],floors[str(meta['m'])]):
                        checks+=1;violations+=int(math.exp(logerr)+1e-9<floor)
                save(path,dict(dataset=dataset,source=source,rep=rep,sampling=sampling,svd=floors,nmf=nmf,methods=rows,seconds=time.perf_counter()-start))
                print(json.dumps(dict(event='matrix_complete',dataset=dataset,source=source,rep=rep,models=len(bank),seconds=time.perf_counter()-start)),flush=True)
    save(P/'results/evaluation_completion.json',dict(seconds=time.perf_counter()-start,models=len(bank),matrix_head_count=64,
         rank_bound_checks=checks,rank_bound_violations=violations))
    assert violations==0

if __name__=='__main__':main()
