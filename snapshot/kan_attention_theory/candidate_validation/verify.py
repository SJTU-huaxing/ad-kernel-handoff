"""Algebra, unseen-input envelope, precision, cone convergence and recurrence."""
import argparse
import torch
from common import *
sys.path.insert(0,str(DEP))
from runtime import causal_prefill,recurrent_step

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--methods',nargs='*');parser.add_argument('--output',default='implementation.json');args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    raw=torch.load(SINGLE/'data/test_000.pt',weights_only=True)
    q,k,v=[raw[s][0,:,:160].cuda().double() for s in ['q','k','v']]
    checks=[]
    for name in args.methods or names():
        if name.startswith(('favor_','aderf_')) and name not in ['favor_1009','aderf_1009']:continue
        m64=Candidate(name);a=m64.features(q,'q');b=m64.features(k,'k')
        matrix=a@b.transpose(-1,-2);w=matrix.tril();ref=w@v/w.sum(-1,keepdim=True)
        out,den,state=causal_prefill(a[:,:128],b[:,:128],v[:,:128])
        pieces=[out]
        for j in range(128,160):
            value,_=recurrent_step(a[:,j:j+1],b[:,j:j+1],v[:,j:j+1],state,optimized=False);pieces.append(value)
        actual=torch.cat(pieces,1)
        err=float((actual-ref).norm()/ref.norm());assert err<1e-7,(name,err)
        m32=Candidate(name,torch.float32);af=m32.features(q.float(),'q');bf=m32.features(k.float(),'k')
        precision=max(float((af.double()-a).norm()/a.norm()),float((bf.double()-b).norm()/b.norm()))
        sliced=Candidate(name,torch.float32,indices=[2,3]);sf=sliced.features(q[2:].float(),'q')
        slicing=float((sf-af[2:]).norm()/af[2:].norm());assert slicing<1e-4,(name,slicing)
        row=dict(method=name,causal_recurrence_relative_l2=err,fp32_feature_relative_l2=precision,
            sliced_heads_relative_l2=slicing,negative_feature_fraction=float(((a<0).sum()+(b<0).sum())/(a.numel()+b.numel())))
        if name=='spectral_pair':
            fq=m64.nystrom(q,'q',32);fk=m64.nystrom(k,'k',32)
            expected=fq@fk.transpose(-1,-2)+m64.t['pair_delta'][:,None,None]*fq[:,:,:1]*fk[:,:,:1].transpose(-1,-2)
            identity=float((matrix-expected).norm()/expected.norm());assert identity<1e-12
            assert (a>=0).all() and (b>=0).all()
            row['pair_algebra_relative_l2']=identity
        checks.append(row);print(json.dumps(row),flush=True)
    cone=[]
    kinds=[x.split('_',1)[1] for x in args.methods if x.startswith('cone_')] if args.methods else ['anchor','mlp','kan','mulkan']
    for kind in kinds:
        model=Candidate('cone_'+kind);qq=q[:,:32];coef=model.features(qq,'q');bb=model.bfeatures(model.t['moment_k'])
        truth=(qq@model.t['moment_k'].transpose(-1,-2)/math.sqrt(128)-model.scale[:,None,None]).exp()
        h=truth@model.basis['weights'];diag=model.basis['diag'];corr=model.basis['correlation'];step=model.basis['lipschitz'][:,None,None]
        z=coef*diag[:,None];hs=h/diag[:,None];grad=z@corr-hs
        pg=(z-(z-grad/step).clamp_min(0)).norm(dim=(-1,-2))/z.norm(dim=(-1,-2))
        loss128=(coef@bb.transpose(-1,-2)-truth).square().mean((-1,-2))
        y=z;alpha=1.
        for _ in range(2048):
            new=(y-(y@corr-hs)/step).clamp_min(0);anew=(1+math.sqrt(1+4*alpha*alpha))/2
            y=new+(alpha-1)/anew*(new-z);z=new;alpha=anew
        lossref=(z/diag[:,None]@bb.transpose(-1,-2)-truth).square().mean((-1,-2))
        cone.append(dict(basis=kind,projected_gradient_relative_norm=pg.tolist(),
            quadratic_risk_128=loss128.tolist(),quadratic_risk_after_2048_more=lossref.tolist(),
            relative_risk_reduction=((loss128-lossref)/loss128).tolist(),
            note='32 unseen queries, training quadrature objective; refinement is diagnostic only and not deployed.'))
    save(P/'checks'/args.output,dict(checks=checks,cone_convergence=cone))

if __name__=='__main__':main()
