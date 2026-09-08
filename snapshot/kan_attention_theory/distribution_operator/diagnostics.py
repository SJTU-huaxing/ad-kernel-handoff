"""Basis-size sensitivity and document-level uncertainty of raw kernel energy."""
import json,math
import torch
from run import P,save,HEADS,RANKS

torch.set_num_threads(4)
package=torch.load(P/'results/internal_moments.pt',weights_only=True)
records=[]
for nn in package['sizes']:
    rec=package['accumulators'][str(nn)];energy=rec['energy']/nn**2
    for h in range(4):
        for rank in [64,128,256]:
            ww=[]
            for side in ['q','k']:
                gram=package[f'prefix_{side}_gram'][str(nn)][h,:rank,:rank]
                ev,u=torch.linalg.eigh(gram);keep=ev>ev.max()*1e-10
                ww.append(u[:,keep]/ev[keep].sqrt()[None])
            cc=ww[0].T@rec['galerkin'][h,:rank,:rank]@ww[1]/nn
            ss=torch.linalg.svdvals(cc);residual=(energy[h]-ss.square().sum())/energy[h]
            assert residual>-1e-7
            bounds={str(m):dict(lower=float(ss[m:].square().sum()/energy[h]),upper=max(0,float((energy[h]-ss[:m].square().sum())/energy[h]))) for m in RANKS}
            records.append(dict(n=nn,head=HEADS[h],basis_dimension=rank,residual_fraction=max(0,float(residual)),bounds=bounds))

gen=torch.Generator().manual_seed(20260911)
indices=torch.randint(256,(2000,256),generator=gen)
weights=torch.stack([torch.bincount(row,minlength=256) for row in indices]).double()
docstats=[]
for nn in package['sizes']:
    dd=package['doc_moments'][str(nn)]['energy'];tokens=nn//256
    for h in range(4):
        matrix=dd[h];total=matrix.sum();estimate=total/nn**2
        boot=(weights@matrix*weights).sum(-1)/nn**2
        ratio=boot/estimate;interval=torch.quantile(ratio,torch.tensor([.025,.5,.975],dtype=torch.float64))
        removed=matrix.sum(0)+matrix.sum(1)-matrix.diag()
        delete=(total-removed)/(nn-tokens)**2/estimate
        offdiag=(total-matrix.trace())/(256*255*tokens*tokens)
        docstats.append(dict(n=nn,head=HEADS[h],same_document_energy_fraction=float(matrix.trace()/total),
            off_diagonal_document_second_moment_ratio=float(offdiag/estimate),
            bootstrap_second_moment_ratio_percentiles=interval.tolist(),
            leave_one_document_out_ratio_range=[float(delete.min()),float(delete.max())],
            max_document_involvement_energy_fraction=float((removed/total).max())))
save(P/'results/distribution_diagnostics.json',dict(basis_size=records,document_sensitivity=docstats,
    bootstrap_draws=2000,bootstrap_unit='Document; identical resampled document weights applied to Q and K marginals.',
    warning='Bootstrap intervals describe this document sample; they are NOT certified population spectral or risk intervals, and cannot cover unseen tails reliably.'))
print(json.dumps(docstats[-4:],indent=2))
