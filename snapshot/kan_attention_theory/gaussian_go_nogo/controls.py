"""CPU controls: finite-sample Gaussian spectra and real matrix-size sensitivity."""
import json
import math
from pathlib import Path
import torch
from run import DATA,LABELS,RANKS

P=Path(__file__).resolve().parent/'results'


def spectrum(q,k):
    z=q@k.T/math.sqrt(128);c=z.max();matrix=(z-c).exp();s=torch.linalg.svdvals(matrix)
    energy=s.square().sum()
    return dict(relative_floor={str(m):float(s[m:].square().sum()/energy) for m in RANKS},
                log_mean_squared_kernel=float(energy.log()+2*c-2*math.log(len(q))))


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    cal=torch.load(P/'calibration.pt',map_location='cpu',weights_only=True)
    gaussian=json.loads((P/'gaussian.json').read_text());valid=[i for i,r in enumerate(gaussian['predictions']) if r['valid']]
    rows=[]
    for n in [256,512,1024]:
        for rep in range(3):
            gen=torch.Generator().manual_seed(838100+n+rep)
            x=torch.randn(len(valid),n,128,dtype=torch.float64,generator=gen)
            y=torch.randn(len(valid),n,128,dtype=torch.float64,generator=gen)
            for j,h in enumerate(valid):
                q=x[j]@cal['q_root'][h];k=y[j]@cal['k_root'][h]
                for mean in ['zero','fitted']:
                    qq=q if mean=='zero' else q+cal['q_mean'][h]
                    kk=k if mean=='zero' else k+cal['k_mean'][h]
                    rows.append(dict(source='matched_gaussian',head=LABELS[h],n=n,rep=rep,mean=mean,**spectrum(qq,kk)))
            print(json.dumps(dict(event='gaussian_control',n=n,rep=rep)),flush=True)
    manifest=json.loads((DATA/'manifest.json').read_text());idx=[manifest['head_labels'].index(h) for h in LABELS]
    qs=[];ks=[]
    for d in manifest['documents']:
        if d['split']!='test':continue
        a=torch.load(DATA/d['file'],map_location='cpu',weights_only=True)
        qs.append(a['q'][idx,512:1024]);ks.append(a['k'][idx,:512])
    qp=torch.cat(qs,1);kp=torch.cat(ks,1)
    for rep in range(4):
        gen=torch.Generator().manual_seed(20260907+rep)
        qi=torch.randperm(qp.shape[1],generator=gen)[:1024];ki=torch.randperm(kp.shape[1],generator=gen)[:1024]
        for h,label in enumerate(LABELS):
            rows.append(dict(source='real_product_size_check',head=label,n=1024,rep=rep,mean='real',
                             **spectrum(qp[h,qi].double(),kp[h,ki].double())))
        print(json.dumps(dict(event='real_size_control',rep=rep)),flush=True)
    (P/'controls.json').write_text(json.dumps(dict(rows=rows,
        note='Finite random matrices are not population spectra, even when data are exactly Gaussian. '
             'Zero-mean Gaussian is a diagnostic changed distribution, not a replacement for the raw real-LLM target.'),indent=2))


if __name__=='__main__':main()
