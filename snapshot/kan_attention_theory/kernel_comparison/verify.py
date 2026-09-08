"""Check independent direct kernels, moment risks, and folded deployed maps."""
import json, math, sys
import numpy as np
import torch
from rf import P,OP,RandomFeatures,bank_features_one_head,a_opt,save
from kernels import GalerkinFeatures,PartitionFeatures
from kernel import ConditionalMeanKernel
from run import profiles,DATA

torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
manifest=json.loads((DATA/'manifest.json').read_text())
entry=next(r for r in manifest['shards'] if r['split']=='test')
raw=torch.load(DATA/entry['file'],weights_only=True)
q=raw['q'][:1,:,512:].permute(1,0,2,3).flatten(1,2).cuda()
k=raw['k'][:1,:,:512].permute(1,0,2,3).flatten(1,2).cuda()
models=torch.load(P/'results/rf_models.pt',weights_only=True)
f,shift=bank_features_one_head(q[0,:64],'q',models,0)
g,kshift=bank_features_one_head(k[0,:64],'k',models,0)
target=q[0,:64].double()@k[0,:64].double().T/math.sqrt(128)
scale=target.max();target=(target-scale).exp();energy=target.square().sum()
max_log_feature_error=0.;max_risk_error=0.
for i,row in enumerate(models):
    original=RandomFeatures(row,128)
    true=original.log_features(q[:,:64],'q')[0]
    reconstructed=f[:,i*128:(i+1)*128].log()+shift[i]
    err=(true-reconstructed).abs().max().item();max_log_feature_error=max(err,max_log_feature_error)
    assert err<1e-9,err
    for m in [16,32,64,128]:
        ff=f[:,i*128:i*128+m];gg=g[:,i*128:i*128+m]
        restore=(shift[i]+kshift[i]-scale+math.log(128/m)).exp()
        pred=ff@gg.T*restore
        direct=(pred-target).square().sum()/energy
        cross=(ff*(target@gg)).sum()*restore
        norm=((ff.T@ff)*(gg.T@gg)).sum()*restore.square()
        moment=(energy-2*cross+norm)/energy
        err=abs(float((moment-direct)/direct.clamp_min(1e-15)))
        max_risk_error=max(max_risk_error,err);assert err<1e-9,err

basis={key:value.cuda() if torch.is_tensor(value) else value for key,value in torch.load(OP/'results/train_basis.pt',weights_only=True).items()}
gp=torch.load(OP/'results/train_galerkin_kernel.pt',weights_only=True)
results=[]
for m in [16,32,64,128]:
    old=ConditionalMeanKernel(m);new=PartitionFeatures(m)
    gal=GalerkinFeatures(m);max_relative=0.
    for x,side in [(q,'q'),(k,'k')]:
        assert torch.equal(old.cells(x,side),new.cells(x,side)),(m,side)
        assert torch.allclose(old.features(x,side),new.features(x,side),rtol=1e-12,atol=0.)
        fs=profiles(x,side,basis)
        expected=torch.stack([fs[h]@r['w'+side]@(r['u' if side=='q' else 'v'][:,:m].cuda()*r['s'][:m].sqrt().cuda()[None]) for h,r in enumerate([{key:value.cuda() if torch.is_tensor(value) else value for key,value in x.items()} for x in gp])])
        expected*=basis['scale'].exp().sqrt()[:,None,None]
        actual=gal.features(x,side)
        relative=((actual-expected).square().sum()/expected.square().sum()).sqrt().item()
        max_relative=max(max_relative,relative);assert relative<1e-8,relative
    results.append(dict(m=m,partition_routing_exact=True,galerkin_feature_relative_l2=max_relative))

# Independent scalar Gaussian integration of dense exponential feature identity.
x,w=np.polynomial.hermite.hermgauss(160);x*=math.sqrt(2);w/=math.sqrt(math.pi)
gh=[]
for lam in [.01,.4,2.,5.]:
    a=float(a_opt(torch.tensor(lam,dtype=torch.float64)))
    for q0,k0 in [(.2,.7),(-.4,.5)]:
        integrand=np.exp(.5*math.log1p(-4*a)+2*a*x*x+math.sqrt(1-4*a)*x*(q0+k0)-.5*(q0*q0+k0*k0))
        err=abs(float((w*integrand).sum()/math.exp(q0*k0)-1))
        assert err<1e-8;gh.append(err)
save(P/'checks/implementation.json',dict(rf_direct_moment_risk_checks=80,
    max_log_feature_error=max_log_feature_error,max_relative_moment_risk_error=max_risk_error,
    deployed_maps=results,max_independent_gaussian_quadrature_error=max(gh)))
print(json.dumps(dict(status='passed',max_relative_moment_risk_error=max_risk_error,deployed_maps=results)),flush=True)
