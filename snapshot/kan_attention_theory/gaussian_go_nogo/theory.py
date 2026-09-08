"""Exact centered Gaussian product-kernel spectrum, with a strict HS-domain check."""
import heapq
import math
import numpy as np

RANKS=[16,32,64,128]


def gaussian_prediction(couplings,log_amplitude=0.,ranks=RANKS):
    a=np.asarray(couplings,dtype=np.float64)
    if np.max(a)>=.5:
        return dict(valid=False,max_coupling=float(a.max()),reason='E[kappa^2] diverges for this unbounded Gaussian surrogate; require max(a)<1/2.',
                    relative_floor=None,log_absolute_floor=None)
    rho=2*a/(1+np.sqrt(1-4*a*a));z=rho*rho
    active=z[z>0];logz=np.log(active);base=np.log1p(-active).sum()
    initial=(0,)*len(active);heap=[(-base,initial)];seen={initial};captured=np.longdouble(0)
    relative={};modes=[]
    for rank in range(1,max(ranks)+1):
        if not heap:
            if rank in ranks:relative[str(rank)]=0.
            continue
        negative,alpha=heapq.heappop(heap);logweight=-negative
        captured+=np.exp(np.longdouble(logweight));modes.append(logweight)
        if rank in ranks:relative[str(rank)]=float(max(np.longdouble(0),1-captured))
        for axis in range(len(active)):
            nxt=list(alpha);nxt[axis]+=1;nxt=tuple(nxt)
            if nxt not in seen:
                seen.add(nxt);heapq.heappush(heap,(-(logweight+logz[axis]),nxt))
    logenergy=-.5*np.log1p(-4*a*a).sum()+2*log_amplitude
    return dict(valid=True,max_coupling=float(a.max()),rho=rho.tolist(),relative_floor=relative,
                log_absolute_floor={m:(float(logenergy+math.log(v)) if v>0 else None) for m,v in relative.items()},
                log_HS_energy=float(logenergy),log_amplitude=float(log_amplitude),top128_normalized_squared_singular_logweights=modes)


def verify():
    from numpy.polynomial.hermite_e import hermegauss
    x,w=hermegauss(120);w=w/math.sqrt(2*math.pi);checks=[]
    for a,b,c in [(.1,0.,0.),(.3,0.,0.),(.45,0.,0.),(.2,.3,-.2)]:
        matrix=np.exp(a*np.outer(x,x)+b*x[:,None]+c*x[None,:])*np.sqrt(w[:,None]*w[None,:])
        observed=np.linalg.svd(matrix,compute_uv=False)[:8]
        rho=2*a/(1+math.sqrt(1-4*a*a))
        l=np.array([b,c]);M=np.array([[1.,-2*a],[-2*a,1.]])
        logamp=l@np.linalg.solve(M,l)
        predicted=math.exp(logamp)*math.sqrt(1+rho*rho)*rho**np.arange(8)
        error=float(np.max(np.abs(observed-predicted)));assert error<2e-10
        checks.append(dict(a=a,b=b,c=c,max_absolute_error=error))
    # Product enumeration reproduces direct sorted 2D geometric weights.
    a=np.array([.15,.3]);p=gaussian_prediction(a);rho=np.array(p['rho']);weights=(1-rho[0]**2)*(1-rho[1]**2)*rho[0]**(2*np.arange(128)[:,None])*rho[1]**(2*np.arange(128)[None,:])
    direct=np.sort(weights.flatten())[::-1]
    for m in RANKS:assert abs(p['relative_floor'][str(m)]-float(direct[m:].sum()))<2e-15
    assert not gaussian_prediction([.5,.1])['valid']
    assert not gaussian_prediction([.8,.1])['valid']
    return dict(gauss_hermite=checks,multidimensional_mode_ordering=True,invalid_domain_rejected=True)


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2))
