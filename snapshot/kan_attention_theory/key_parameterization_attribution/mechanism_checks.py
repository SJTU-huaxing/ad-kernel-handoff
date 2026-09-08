"""Analytical local log-feature derivatives, checked without data fitting."""
from core_attribution import *

def main():
    torch.manual_seed(705);torch.set_num_threads(4)
    q=torch.randn(64,dtype=torch.float64).softmax(-1)
    target=torch.randn(17,dtype=torch.float64).softmax(-1)
    rows=[]
    for kind in ['ad','exp']:
        out=torch.randn(17,64,dtype=torch.float64,requires_grad=True)
        if kind=='ad':
            z=torch.cat([out[:,:63],torch.zeros_like(out[:,-1:])],-1)
            pi=z.softmax(-1);logf=z.log_softmax(-1)+out[:,-1:]
        else:
            pi=out.softmax(-1);logf=out
        logkernel=(logf+q.log()[None]).logsumexp(-1)
        lp=logkernel.log_softmax(-1);p=lp.exp()
        loss=(target*(target.log()-lp)).sum()
        grad=torch.autograd.grad(loss,out,retain_graph=True)[0]
        responsibility=(logf+q.log()[None]-logkernel[:,None]).exp()
        residual=(p-target)[:,None]
        expected=(torch.cat([residual*(responsibility[:,:63]-pi[:,:63]),residual],-1)
                  if kind=='ad' else residual*responsibility)
        error=float((grad-expected).abs().max().detach());assert error<1e-12
        mass=logf.logsumexp(-1)
        massgrad=torch.autograd.grad(mass.sum(),out)[0]
        mg=torch.zeros_like(out) if kind=='ad' else pi
        if kind=='ad':mg[:,-1]=1
        mgerr=float((massgrad-mg).abs().max().detach());assert mgerr<1e-12
        rows.append(dict(kind=kind,kl_gradient_max_error=error,log_mass_gradient_max_error=mgerr))
    save(P/'checks/mechanism.json',dict(passed=True,rows=rows,
        scope='Exact coordinate derivative identities, not a proof of optimizer superiority or finite-network capacity equivalence.'))
    print(json.dumps(rows),flush=True)

if __name__=='__main__':main()
