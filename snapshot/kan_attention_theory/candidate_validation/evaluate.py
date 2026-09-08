"""Document-disjoint empirical-product risks; raw kernel never normalized by row."""
import argparse,gc,time
import torch
from common import *

@torch.no_grad()
def validation_data():
    manifest=json.loads((SINGLE/'data/manifest.json').read_text())
    raw=[torch.load(SINGLE/'data'/r['file'],weights_only=True) for r in manifest['shards'] if r['split']=='validation']
    gen=torch.Generator().manual_seed(20260917);out={}
    for side in ['q','k']:
        x=torch.cat([b[side] for b in raw]);x=x[:,:,512:] if side=='q' else x[:,:,:512]
        perm=torch.stack([torch.randperm(512,generator=gen) for _ in range(len(x))])
        x=x.gather(2,perm[:,None,:,None].expand(-1,4,-1,128))
        out[side]=x.permute(1,2,0,3).flatten(1,2).contiguous().cuda()
    return out

@torch.no_grad()
def make_features(ds,data,n,methods):
    out={};started=time.perf_counter()
    for method in methods:
        path=P/f'results/features_{ds}_{n}_{method}.pt'
        if path.exists():out[method]=torch.load(path,weights_only=True);continue
        model=Candidate(method);box={}
        for side in ['q','k']:
            blocks=[]
            for start in range(0,n,1024):blocks.append(model.features(data[side][:,start:min(n,start+1024)],side).cpu())
            box[side]=torch.cat(blocks,1)
            assert torch.isfinite(box[side]).all(),(ds,method,side)
        torch.save(box,path);out[method]=box
        print(json.dumps(dict(event='features',dataset=ds,n=n,method=method,seconds=time.perf_counter()-started)),flush=True)
        del model;gc.collect()
    return out

@torch.no_grad()
def evaluate(ds,data,n,methods):
    started=time.perf_counter();features=make_features(ds,data,n,methods)
    scale=torch.load(P/'results/construction.pt',weights_only=True)['scale'].cuda()
    output=P/f'results/kernel_{ds}_{n}.json'
    previous=[r for r in json.loads(output.read_text())['results'] if r['method'] not in methods] if output.exists() else []
    rows=[];gen=torch.Generator().manual_seed(20260918)
    iq=torch.randint(n,(262144,),generator=gen).cuda();ik=torch.randint(n,(262144,),generator=gen).cuda()
    for h,head in enumerate(HEADS):
        qs=data['q'][h,:n].double();ks=data['k'][h,:n].double()
        fq={name:features[name]['q'][h].cuda() for name in methods}
        fk={name:features[name]['k'][h].cuda() for name in methods}
        widths=[fq[name].shape[-1] for name in methods];fbank=torch.cat([fk[name] for name in methods],-1)
        cross={name:torch.zeros((),device='cuda',dtype=torch.float64) for name in methods}
        captured={name:torch.zeros((),device='cuda',dtype=torch.float64) for name in methods}
        wh={};ranks={}
        for name in methods:
            if not name.startswith(('avg_','cone_')):continue
            b=fk[name];diag=b.square().mean(0).sqrt();g=(b/diag).T@(b/diag)/n
            s,u=torch.linalg.eigh(g);keep=s>s[-1]*1e-10
            wh[name]=u[:,keep]/s[keep].sqrt()[None]/diag[:,None];ranks[name]=int(keep.sum())
        energy=torch.zeros((),device='cuda',dtype=torch.float64);mass=energy.clone()
        for start in range(0,n,2048):
            truth=(qs[start:start+2048]@ks.T/math.sqrt(128)-scale[h]).exp()
            energy+=truth.square().sum();mass+=truth.sum()
            pieces=(truth@fbank).split(widths,-1)
            for name,x in zip(methods,pieces):
                cross[name]+=(fq[name][start:start+2048]*x).sum()
                if name in wh:captured[name]+=((x/n)@wh[name]).square().sum()/n
        target_log=(qs[iq]*ks[ik]).sum(-1)/math.sqrt(128)-scale[h];target=target_log.exp()
        for name in methods:
            gq=fq[name].T@fq[name];gk=fk[name].T@fk[name]
            pred2=(gq*gk).sum();risk=(energy-2*cross[name]+pred2)/energy
            assert risk>=-1e-7,(name,risk)
            sample=(fq[name][iq]*fk[name][ik]).sum(-1)
            positive=sample>0;idiv=None;logmae=None
            if positive.all():
                logp=sample.log();idiv=float((sample-target+target*(target_log-logp)).sum()/target.sum())
                logmae=float((logp-target_log).abs().mean())
            row=dict(dataset=ds,n=n,head=head,method=name,m=widths[methods.index(name)],
                kernel_nmse=max(0,float(risk)),sample_normalized_idiv=idiv,sample_log_mae=logmae,
                sample_negative_fraction=float((sample<0).double().mean()),sample_zero_fraction=float((sample==0).double().mean()),
                sampled_pairs=262144,log_squared_energy_mean=float((energy/n**2).log()+2*scale[h]),
                predicted_energy_ratio=float(pred2/energy))
            if name in wh:
                span=1-captured[name]/(energy/n**2)
                row.update(span_risk=float(span),coefficient_excess=float(risk-span),retained_basis_rank=ranks[name],
                    decomposition_note='Heldout span projection is diagnostic only. Coefficient excess includes positivity, train quadrature, and numerical solve errors; not a positivity-only lower bound.')
            rows.append(row)
        save(output,dict(results=previous+rows,seconds_current_stage=time.perf_counter()-started,current_stage_methods=methods,
            target='Raw exp(q.k/sqrt(128)); exact full product relative squared risk, plus fixed independent pair Monte Carlo KL/log metrics.',
            split=ds,empirical_only=True,selection='All requested frozen candidates; no test fit.'))
        print(json.dumps(dict(event='kernel_head',dataset=ds,n=n,head=head,seconds=time.perf_counter()-started,
            risk={r['method']:r['kernel_nmse'] for r in rows if r['head']==head})),flush=True)
        del fbank,fq,fk,truth,pieces
    return rows

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--datasets',nargs='+',default=['validation','moment_train','internal','official'])
    parser.add_argument('--n',type=int,default=8192);parser.add_argument('--methods',nargs='*');args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    data=load_data();data['validation']=validation_data();methods=args.methods or names()
    for ds in args.datasets:
        n=10240 if ds=='official' and args.n==8192 else min(args.n,data[ds]['q'].shape[1])
        evaluate(ds,data[ds],n,methods)

if __name__=='__main__':main()
