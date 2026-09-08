"""Independent algebra and quadrature checks for the exact raw-kernel construction."""
import json, math
import numpy as np
import torch
from experiment import P, Features, a_opt, save

torch.set_num_threads(2)
cal={k:v.cuda() for k,v in torch.load(P/'results/calibration.pt',weights_only=True).items()}
torch.manual_seed(99)
q=torch.randn(4,19,128,device='cuda',dtype=torch.float64)*.3+cal['q_mean'][:,None]
k=torch.randn_like(q)*.3+cal['k_mean'][:,None]
errors={}
for method in ['favor_orf','sderf_orf','balanced_orf','balanced_sderf_sobol']:
    f=Features(cal,32,11,method)
    zq,bq=f.transform(q,'q');zk,bk=f.transform(k,'k')
    exact=(q*k).sum(-1)/math.sqrt(128)
    restored=(zq*zk).sum(-1)+bq+bk+.5*(zq.square().sum(-1)+zk.square().sum(-1))
    error=(restored-exact).abs().max().item();assert error<1e-8
    lf=f.log_feature(q,'q');lg=f.log_feature(k,'k')
    p=(lf+lg).logsumexp(-1).exp();direct=(lf.exp()*lg.exp()).sum(-1)
    relative=((p-direct).abs()/p).max().item();assert relative<1e-10
    errors[method]=dict(dot_identity_error=error,log_feature_product_relative_error=relative)
# This integration is independent of the 128-dimensional implementation.
u,w=np.polynomial.hermite.hermgauss(160);u*=math.sqrt(2);w/=math.sqrt(math.pi)
gh=[]
for lam in [.01,.4,2.,5.]:
    a=float(a_opt(torch.tensor(lam,dtype=torch.float64)));b=math.sqrt(1-4*a);ld=.25*math.log(1-4*a)
    for x,y in [(.2,.7),(-.4,.5),(.6,.6)]:
        estimate=np.sum(w*np.exp(2*ld+2*a*u*u+b*u*(x+y)-(x*x+y*y)/2))
        err=abs(estimate/math.exp(x*y)-1);assert err<1e-8,(lam,err)
        gh.append(dict(lambda_sum=lam,x=x,y=y,relative_error=err))
pred=torch.tensor([-.4,1.3],dtype=torch.float64,requires_grad=True)
gradient=torch.autograd.grad((pred.exp()-pred.detach().exp()*pred).sum(),pred)[0]
assert gradient.abs().max()==0
beta=1.5
pp=torch.tensor([-.4,1.3],dtype=torch.float64,requires_grad=True)
bb=((beta-1)*(beta*pp).exp()-beta*(pp.detach()+(beta-1)*pp).exp())/(beta*(beta-1))
beta_gradient=torch.autograd.grad(bb.sum(),pp)[0]
assert beta_gradient.abs().max()<1e-12
yy=np.array([.03,1.,5.]);yh=np.array([2.,1.2,.02])
direct=(yy**beta+(beta-1)*yh**beta-beta*yy*yh**(beta-1))/(beta*(beta-1))
factored=(2/3)*(np.sqrt(yh)-np.sqrt(yy))**2*(np.sqrt(yh)+2*np.sqrt(yy))
assert np.max(np.abs(direct-factored))<1e-12
torch.manual_seed(101)
xx=torch.randn(9,3,dtype=torch.float64)*.2;yy=torch.randn_like(xx)*.2
basis=torch.zeros(6,3,3,dtype=torch.float64)
for j,(i,k) in enumerate([(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]):
    basis[j,i,k]=basis[j,k,i]=1 if i==k else 1/math.sqrt(2)
theta=torch.tensor([1.2,1.3,1.4,.1,-.1,.15],dtype=torch.float64,requires_grad=True)
def variance_objective(z):
    h=torch.einsum('j,jab->ab',z,basis)
    exponent=(xx*(xx@h)).sum(-1)+(yy*torch.linalg.solve(h,yy.T).T).sum(-1)+4*(xx*yy).sum(-1)
    return exponent.exp().mean()
hessian=torch.autograd.functional.hessian(variance_objective,theta)
mineig=torch.linalg.eigvalsh(hessian).min().item();assert mineig>=-1e-10
save(P/'results/method_checks.json',dict(feature_checks=errors,independent_gauss_hermite=gh,
    idiv_stationary_at_true_raw_kernel=True,beta15_stationary_at_true_raw_kernel=True,
    beta15_nonnegative_factorization_verified=True,restricted_variance_hessian_min_eigenvalue=mineig))
print(json.dumps(dict(feature_checks=errors,max_gh_relative_error=max(r['relative_error'] for r in gh))))
