"""Frozen real-LLM Go/No-Go: Gaussian population floor, finite SVD, NMF and FAVOR+."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import torch
from theory import RANKS,gaussian_prediction,verify

P=Path(__file__).resolve().parent
DATA=P.parent/'real_llm_pilot/data_qwen25_1p5b'
LABELS=[[l,h] for l in [0,7,14,27] for h in [0,3,6,9]]


def save(name,obj):
    (P/name).write_text(json.dumps(obj,indent=2,allow_nan=False))


@torch.no_grad()
def load_pools():
    manifest=json.loads((DATA/'manifest.json').read_text());indices=[manifest['head_labels'].index(h) for h in LABELS];pools={}
    for split in ['train','test']:
        docs=[d for d in manifest['documents'] if d['split']==split]
        q=[];k=[]
        for d in docs:
            b=torch.load(DATA/d['file'],map_location='cpu',weights_only=True)
            q.append(b['q'][indices,512:1024]);k.append(b['k'][indices,:512])
        pools[split]=dict(q=torch.stack(q,1).cuda(),k=torch.stack(k,1).cuda(),documents=docs)
    return manifest,pools


def sqrt_matrix(cov):
    s,u=torch.linalg.eigh(cov);return (u*s.clamp_min(0).sqrt()[:,None,:])@u.transpose(-1,-2)


@torch.no_grad()
def covariances(pools):
    histories=[];cal={}
    for n in [16,32,64]:
        for side in ['q','k']:
            x=pools[side][:,:n].flatten(1,2).double();mean=x.mean(1);xc=x-mean[:,None]
            cov=xc.transpose(-1,-2)@xc/(x.shape[1]-1)
            cal[side+'_mean']=mean;cal[side+'_cov']=cov;cal[side+'_root']=sqrt_matrix(cov)
        A=cal['q_root']@cal['k_root']/math.sqrt(128);s=torch.linalg.svdvals(A)
        histories.append(dict(training_documents=n,samples_per_marginal=n*512,max_a=s[:,0].tolist()))
    predictions=[]
    for h,label in enumerate(LABELS):
        b=cal['q_root'][h]@cal['k_mean'][h]/math.sqrt(128)
        c=cal['k_root'][h]@cal['q_mean'][h]/math.sqrt(128)
        amplitude=0.
        if s[h,0]<.5:
            identity=torch.eye(128,device='cuda',dtype=torch.float64)
            M=torch.cat([torch.cat([identity,-2*A[h]],1),torch.cat([-2*A[h].T,identity],1)],0)
            linear=torch.cat([b,c]);constant=(cal['q_mean'][h]*cal['k_mean'][h]).sum()/math.sqrt(128)
            amplitude=float(constant+linear@torch.linalg.solve(M,linear))
        pred=gaussian_prediction(s[h].cpu().numpy(),log_amplitude=amplitude)
        pred.update(head=label,couplings=s[h].tolist(),q_mean_norm=float(cal['q_mean'][h].norm()),k_mean_norm=float(cal['k_mean'][h].norm()))
        predictions.append(pred)
    torch.save({k:v.cpu() for k,v in cal.items()},P/'calibration.pt')
    save('gaussian.json',dict(predictions=predictions,covariance_size_check=histories,
        note='All 32768 train vectors per marginal; covariance centered, raw kernel not centered. '
             'Nonzero Gaussian means change only an overall singular-value factor when HS-valid; relative floors retain covariance-only form.'))
    return cal


@torch.no_grad()
def decompose(q,k):
    logits=q.double()@k.double().transpose(-1,-2)/math.sqrt(128)
    maximum=logits.amax((-2,-1));A=(logits-maximum[:,None,None]).exp()
    norm=A.square().sum((-2,-1)).sqrt();A=A/norm[:,None,None]
    U,S,Vh=torch.linalg.svd(A,full_matrices=False,driver='gesvd')
    floors={str(m):S[:,m:].square().sum(-1).tolist() for m in RANKS}
    logenergy=(2*(maximum+norm.log())).tolist()
    return A,(U,S,Vh),floors,logenergy,logits


@torch.no_grad()
def nmf_start(A,svd,m,kind,seed):
    h,n,_=A.shape;generator=torch.Generator(device='cuda').manual_seed(seed)
    if kind=='random':
        W=torch.rand(h,n,m,device='cuda',dtype=torch.float64,generator=generator)
        H=torch.rand(h,m,n,device='cuda',dtype=torch.float64,generator=generator)
        target=A.mean((-2,-1)).clamp_min(1e-300)
        scale=(target/((W@H).mean((-2,-1)))).sqrt()
        return W*scale[:,None,None],H*scale[:,None,None]
    U,S,Vh=svd;W=torch.zeros(h,n,m,device='cuda',dtype=torch.float64);H=torch.zeros(h,m,n,device='cuda',dtype=torch.float64)
    W[:,:,0]=U[:,:,0].abs()*S[:,0,None].sqrt();H[:,0]=Vh[:,0].abs()*S[:,0,None].sqrt()
    for j in range(1,m):
        u=U[:,:,j];v=Vh[:,j];up=u.clamp_min(0);un=(-u).clamp_min(0);vp=v.clamp_min(0);vn=(-v).clamp_min(0)
        upn=up.norm(dim=-1);unn=un.norm(dim=-1);vpn=vp.norm(dim=-1);vnn=vn.norm(dim=-1)
        positive=upn*vpn>=unn*vnn
        uu=torch.where(positive[:,None],up/upn.clamp_min(1e-300)[:,None],un/unn.clamp_min(1e-300)[:,None])
        vv=torch.where(positive[:,None],vp/vpn.clamp_min(1e-300)[:,None],vn/vnn.clamp_min(1e-300)[:,None])
        amplitude=(S[:,j]*torch.maximum(upn*vpn,unn*vnn)).clamp_min(0).sqrt()
        W[:,:,j]=uu*amplitude[:,None];H[:,j]=vv*amplitude[:,None]
    # Small positive initialization avoids irrevocably locked zero entries.
    floor=(A.mean((-2,-1))/m).sqrt()*1e-3
    W=W+floor[:,None,None];H=H+floor[:,None,None]
    return W,H


@torch.no_grad()
def optimize_nmf(A,W,H,steps):
    history=[];best_error=(A-W@H).square().sum((-2,-1));bestW=W.clone();bestH=H.clone()
    for step in range(1,steps+1):
        H.mul_((W.transpose(-1,-2)@A)/((W.transpose(-1,-2)@W)@H).clamp_min(1e-300))
        W.mul_((A@H.transpose(-1,-2))/(W@(H@H.transpose(-1,-2))).clamp_min(1e-300))
        if step%100==0 or step==steps:
            norms=W.norm(dim=1).clamp_min(1e-150);W/=norms[:,None,:];H*=norms[:,:,None]
            error=(A-W@H).square().sum((-2,-1));improved=error<best_error
            best_error=torch.minimum(error,best_error)
            bestW=torch.where(improved[:,None,None],W,bestW);bestH=torch.where(improved[:,None,None],H,bestH)
            history.append(dict(step=step,error=error.tolist()))
    return bestW,bestH,best_error,history


@torch.no_grad()
def nmf_curve(A,svd,steps):
    records=[];previous=None
    for m in RANKS:
        best=None;starts=[]
        for kind in ['nndsvd','random','nested']:
            if kind=='nested' and previous is None:continue
            if kind=='nested':
                W,H=nmf_start(A,svd,m,'random',991+m);W*=1e-5;H*=1e-5
                W[:,:,:previous[0].shape[-1]]=previous[0];H[:,:previous[1].shape[1]]=previous[1]
            else:W,H=nmf_start(A,svd,m,kind,991+m)
            W,H,error,history=optimize_nmf(A,W,H,steps)
            starts.append(dict(kind=kind,final_error=error.tolist(),history=history))
            if best is None:best=(W,H,error)
            else:
                take=error<best[2]
                best=(torch.where(take[:,None,None],W,best[0]),torch.where(take[:,None,None],H,best[1]),torch.minimum(error,best[2]))
        previous=best[:2]
        records.append(dict(m=m,error=best[2].tolist(),starts=starts))
        print(json.dumps(dict(event='nmf_rank_complete',m=m,median_error=float(best[2].median()))),flush=True)
    return records


def orf(heads,seed):
    generator=torch.Generator(device='cuda').manual_seed(seed)
    raw=torch.randn(heads,128,128,device='cuda',dtype=torch.float64,generator=generator)
    Q,R=torch.linalg.qr(raw);Q=Q*torch.diagonal(R,dim1=-2,dim2=-1).sign()[:,None,:]
    radii=torch.randn(heads,128,128,device='cuda',dtype=torch.float64,generator=generator).norm(dim=-1)
    return Q.transpose(-1,-2)*radii[:,:,None]


def log_sse(logpred,logtarget):
    difference=(logpred-logtarget).abs()
    logabs=torch.maximum(logpred,logtarget)+(-torch.expm1(-difference)).log()
    return (2*logabs).flatten(1).logsumexp(-1)


@torch.no_grad()
def favor(q,k,logits,cal,seeds):
    records=[];d=128;logenergy=(2*logits).flatten(1).logsumexp(-1)
    for seed in seeds:
        omega=orf(len(q),seed)
        for centered in [False,True]:
            qq=q.double()-cal['q_mean'][:,None] if centered else q.double()
            kk=k.double()-cal['k_mean'][:,None] if centered else k.double()
            lq=qq@omega.transpose(-1,-2)*d**(-.25)-qq.square().sum(-1,keepdim=True)/(2*math.sqrt(d))
            lk=kk@omega.transpose(-1,-2)*d**(-.25)-kk.square().sum(-1,keepdim=True)/(2*math.sqrt(d))
            if centered:
                common=(cal['q_mean']*cal['k_mean']).sum(-1)/(2*math.sqrt(d))
                lq+=((qq*cal['k_mean'][:,None]).sum(-1)/math.sqrt(d)+common[:,None])[:,:,None]
                lk+=((kk*cal['q_mean'][:,None]).sum(-1)/math.sqrt(d)+common[:,None])[:,:,None]
            for m in RANKS:
                # Exact log-sum-exp of unnormalized features, no row shift or epsilon changes.
                logerrors=[]
                for start in range(0,q.shape[1],32):
                    lp=(lq[:,start:start+32,None,:m]+lk[:,None,:,:m]).logsumexp(-1)-math.log(m)
                    logerrors.append(log_sse(lp,logits[:,start:start+32]))
                logerror=torch.stack(logerrors).logsumexp(0)-logenergy
                records.append(dict(seed=seed,method='centered_FAVOR_plus' if centered else 'FAVOR_plus',m=m,
                    log_nmse=logerror.tolist(),nmse=[math.exp(x) if x<700 else None for x in logerror.tolist()]))
    return records


@torch.no_grad()
def main():
    global P
    parser=argparse.ArgumentParser();parser.add_argument('--nmf-steps',type=int,default=800)
    parser.add_argument('--out',type=Path,default=P/'results')
    parser.add_argument('--repetitions',type=int,default=4);parser.add_argument('--favor-seeds',type=int,default=10)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    P=args.out;P.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    save('theory_checks.json',verify());start=time.perf_counter()
    manifest,pools=load_pools();cal=covariances(pools['train'])
    seeds=[1009+37*i for i in range(args.favor_seeds)]
    save('protocol.json',dict(model=manifest['model'],model_revision=manifest['revision'],head_labels=LABELS,ranks=RANKS,
        train_documents=[d['file'] for d in pools['train']['documents']],test_documents=[d['file'] for d in pools['test']['documents']],
        samples_per_covariance=32768,matrix_size=512,repetitions=args.repetitions,nmf_steps=args.nmf_steps,
        favor_seeds=seeds,q_positions=[512,1023],k_positions=[0,511],
        primary='Independent Q and K draws from held-out marginal pools; empirical product measure.',
        secondary='Same-document legal rectangular blocks; different distribution from product of global marginals.',
        nmf='Transductive Euclidean NMF, independent nonnegative W/H; best of NNDSVD, random and nested starts; not a neural model.',
        baseline='FAVOR+ Gaussian orthogonal directions with chi_d radii; centered variant restores all linear and constant terms.',
        metric='Relative raw kernel squared error, with log absolute MSE also available; no row normalization.',
        data_note='Existing cached WikiText-2 article prefixes, official source train/test splits; selected subsets, not full benchmark.'))
    qp=pools['test']['q'].flatten(1,2);kp=pools['test']['k'].flatten(1,2)
    for source in ['product','same_context']:
        for rep in range(args.repetitions):
            path=P/f'{source}_{rep}.json'
            if path.exists():continue
            generator=torch.Generator().manual_seed(20260907+rep)
            if source=='product':
                qi=torch.randperm(qp.shape[1],generator=generator)[:512];ki=torch.randperm(kp.shape[1],generator=generator)[:512]
                q=qp[:,qi];k=kp[:,ki]
                sampling=dict(q_indices=qi.tolist(),k_indices=ki.tolist())
            else:q=pools['test']['q'][:,rep];k=pools['test']['k'][:,rep];sampling=dict(document=pools['test']['documents'][rep]['file'])
            A,svd,floors,energy,logits=decompose(q,k)
            print(json.dumps(dict(event='svd_complete',source=source,rep=rep,seconds=time.perf_counter()-start)),flush=True)
            nmf=nmf_curve(A,svd,args.nmf_steps)
            random=favor(q,k,logits,cal,seeds)
            record=dict(source=source,rep=rep,sampling=sampling,svd=floors,log_energy=energy,nmf=nmf,favor=random,
                logit_ranges=dict(min=logits.amin((-2,-1)).tolist(),max=logits.amax((-2,-1)).tolist()),
                scaled_kernel_underflows=(A==0).sum((-2,-1)).tolist(),seconds=time.perf_counter()-start)
            for row in nmf:
                assert all(e+1e-10>=b for e,b in zip(row['error'],floors[str(row['m'])]))
            save(path.name,record)
            torch.save(dict(W=None if args.smoke else 'NMF factors not exported; optimization histories and achieved errors retained.',
                            singular_values=svd[1].cpu()),P/f'{source}_{rep}_spectrum.pt')
            print(json.dumps(dict(event='matrix_complete',source=source,rep=rep,seconds=time.perf_counter()-start)),flush=True)
            if args.smoke:return
    save('completion.json',dict(seconds=time.perf_counter()-start,matrices=args.repetitions*2*16))


if __name__=='__main__':main()
