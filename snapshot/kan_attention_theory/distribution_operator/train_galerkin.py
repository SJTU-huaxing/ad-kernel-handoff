"""Schmidt truncation of a TRAIN-distribution Galerkin operator, then frozen eval.

This is a signed reference, not the nonnegative conditional-mean construction.
"""
import json,math
import torch
from run import P,HEADS,RANKS,save,tensor_save

torch.set_num_threads(4)
train=torch.load(P/'results/moment_train_moments.pt',weights_only=True);n=train['sizes'][-1]
ac=train['accumulators'][str(n)];rq=train['q_r'];rk=train['k_r'];models=[]
for h in range(4):
    transforms=[];checks=[]
    for r in [rq[h],rk[h]]:
        col=r.norm(dim=0).clamp_min(1e-300);scaled=r/col[None]
        u,s,vh=torch.linalg.svd(scaled,full_matrices=False)
        keep=s>s[0]*1e-4
        w=(math.sqrt(n)/col[:,None])*vh.T[:,keep]/s[keep][None]
        ortho=r@w/math.sqrt(n)
        error=(ortho.T@ortho-torch.eye(int(keep.sum()))).abs().max().item()
        assert error<1e-7,error
        transforms.append(w);checks.append(dict(dimensions=int(keep.sum()),orthogonality_error=error))
    aq=rq[h]@transforms[0];ak=rk[h]@transforms[1]
    c=aq.T@ac['galerkin'][h]@ak/n**2
    u,s,vh=torch.linalg.svd(c,full_matrices=False)
    models.append(dict(wq=transforms[0],wk=transforms[1],u=u,s=s,v=vh.T,checks=checks))
tensor_save(P/'results/train_galerkin_kernel.pt',models)
results=[];violations=0;max_train_identity=0.
for ds in ['moment_train','internal','official','disjoint_ab','disjoint_ba']:
    package=torch.load(P/f'results/{ds}_moments.pt',weights_only=True)
    bounds=json.loads((P/f'results/{ds}_summary.json').read_text())
    for ordinal,nn in enumerate(package['sizes']):
        ac=package['accumulators'][str(nn)]
        for h,model in enumerate(models):
            aq=package['q_r'][h]@model['wq'];ak=package['k_r'][h]@model['wk']
            gq=aq.T@package['prefix_q_gram'][str(nn)][h]@aq/nn
            gk=ak.T@package['prefix_k_gram'][str(nn)][h]@ak/nn
            cross=aq.T@ac['galerkin'][h]@ak/nn**2;energy=ac['energy'][h]/nn**2
            for m in RANKS:
                core=(model['u'][:,:m]*model['s'][:m][None])@model['v'][:,:m].T
                pred2=(core*(gq@core@gk)).sum();joint=(core*cross).sum()
                risk=(energy-2*joint+pred2)/energy
                assert torch.isfinite(risk) and risk>=-1e-7
                if ds=='moment_train' and nn==n:
                    expected=(energy-model['s'][:m].square().sum())/energy
                    max_train_identity=max(max_train_identity,abs(float(risk-expected)))
                lower=bounds[ordinal]['heads'][h]['signed_floor_bracket'][str(m)]['lower']
                violations+=int(float(risk)+1e-7<lower)
                results.append(dict(dataset=ds,n=nn,head=HEADS[h],m=m,nmse=max(0,float(risk)),
                    retained_dimensions=[len(model['wq'][0]),len(model['wk'][0])]))
assert violations==0 and max_train_identity<1e-7
save(P/'results/train_galerkin_results.json',dict(results=results,normalization=[r['checks'] for r in models],
    lower_bound_violations=violations,max_training_projection_identity_error=max_train_identity,
    note='Only training moments determine functions, coefficients, and singular truncation; signed features. No held-out fitting.'))
for r in results:
    if r['dataset']=='internal' and r['n']==131072 and r['m']==64:print(r)
