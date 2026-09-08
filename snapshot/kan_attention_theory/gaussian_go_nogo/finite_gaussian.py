"""Finite Gaussian surrogate matrices, including heads with no HS population kernel."""
import json
from pathlib import Path
import torch
from controls import spectrum
from run import LABELS

P=Path(__file__).resolve().parent/'results'


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    cal=torch.load(P/'calibration.pt',map_location='cpu',weights_only=True)
    pred=json.loads((P/'gaussian.json').read_text())['predictions']
    rows=[r for r in json.loads((P/'controls.json').read_text())['rows']
          if r['source']=='matched_gaussian' and r['mean']=='fitted' and r['n'] in [512,1024]]
    invalid=[h for h,p in enumerate(pred) if not p['valid']]
    for n in [512,1024]:
        for rep in range(3):
            generator=torch.Generator().manual_seed(747900+n+rep)
            for h in invalid:
                q=torch.randn(n,128,dtype=torch.float64,generator=generator)@cal['q_root'][h]+cal['q_mean'][h]
                k=torch.randn(n,128,dtype=torch.float64,generator=generator)@cal['k_root'][h]+cal['k_mean'][h]
                rows.append(dict(source='matched_gaussian',head=LABELS[h],n=n,rep=rep,mean='fitted',**spectrum(q,k)))
            print(json.dumps(dict(n=n,rep=rep,complete_rows=len(rows))),flush=True)
    assert len(rows)==96
    (P/'finite_gaussian.json').write_text(json.dumps(dict(rows=rows,
        note='This is an empirical finite-matrix surrogate prediction, not the Gaussian population Schmidt floor. '
             'Every finite draw exists even for HS-invalid population heads; finite error must not be reported as a finite population bound.'),indent=2))


if __name__=='__main__':main()
