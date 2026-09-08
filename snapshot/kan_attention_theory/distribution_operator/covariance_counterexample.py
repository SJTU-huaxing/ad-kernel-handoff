"""Same bounded covariance, radically different true softmax-kernel spectra."""
import math
import torch
from run import P,save

a=.2;rho=2*a/(1+math.sqrt(1-4*a*a));rows=[]
for eps in [1.,.1,.01]:
    x=torch.tensor([0.,-math.sqrt(a/eps),math.sqrt(a/eps)],dtype=torch.float64)
    p=torch.tensor([1-eps,eps/2,eps/2],dtype=torch.float64)
    K=torch.exp(x[:,None]*x[None])*torch.sqrt(p[:,None]*p[None])
    s=torch.linalg.svdvals(K);energy=K.square().sum();formula=1-eps**2+eps**2*math.cosh(2*a/eps)
    assert abs(float((p*x).sum()))<1e-12 and abs(float((p*x*x).sum())-a)<1e-12
    assert abs(float(energy)/formula-1)<1e-12
    rows.append(dict(epsilon=eps,variance=a,support=x.tolist(),kernel_second_moment=float(energy),
        rank1_relative_floor=float(s[1:].square().sum()/energy),singular_values=s.tolist()))
save(P/'checks/covariance_counterexample.json',dict(gaussian_variance=a,gaussian_HS_valid=True,
    gaussian_second_moment=1/math.sqrt(1-4*a*a),gaussian_rank1_relative_floor=rho**2,discrete_distributions=rows))
print(rows)
