"""CPU-only paired-kernel tail analysis and empirical rectangular SVD oracle."""
import json
import math
from pathlib import Path
import torch

P=Path(__file__).resolve().parent


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    manifest=json.loads((P/'data/manifest.json').read_text());results={}
    for split in ['train','validation','test']:
        logits=[]
        generator=torch.Generator().manual_seed(20260906+(10000 if split=='validation' else 20000))
        for shard in manifest['shards']:
            if shard['split']!=split:continue
            batch=torch.load(P/'data'/shard['file'],weights_only=True,map_location='cpu')
            q=batch['q'].float();k=batch['k'].float()
            if split!='train':
                permutations=torch.stack([torch.randperm(512,generator=generator) for _ in range(len(q))])
                q=q[:,:,512:];k=k[:,:,:512].gather(2,permutations[:,None,:,None].expand(-1,4,-1,128))
            logits.append((q*k).sum(-1)/math.sqrt(128))
        z=torch.cat(logits).permute(1,0,2).flatten(1).double();rows=[]
        for h,label in enumerate(manifest['head_labels']):
            x=z[h];n=len(x);row=dict(head=label,pairs=n,logit_mean=float(x.mean()),logit_std=float(x.std()),
                logit_quantiles=dict(zip(['min','1%','50%','99%','99.9%','max'],
                                  x.quantile(torch.tensor([0.,.01,.5,.99,.999,1.],dtype=torch.float64)).tolist())))
            y=(x-x.max()).exp();order=y.sort(descending=True).values
            for fraction in [.0001,.001,.01]:
                count=math.ceil(n*fraction)
                row[f'top_{fraction}_mass_share']=float(order[:count].sum()/order.sum())
                row[f'top_{fraction}_energy_share']=float(order[:count].square().sum()/order.square().sum())
            rows.append(row)
        results[split]=rows
    batch=torch.load(P/'data/test_000.pt',weights_only=True,map_location='cpu');spectra=[]
    for i in range(32):
        q=batch['q'][i,:,512:].double();k=batch['k'][i,:,:512].double()
        z=q@k.transpose(-1,-2)/math.sqrt(128);c=z.amax((-2,-1));kernel=(z-c[:,None,None]).exp()
        sv=torch.linalg.svdvals(kernel);energy=sv.square().sum(-1)
        spectra.append(dict(ordinal=i,log_energy=(energy.log()+2*c).tolist(),
            residuals={str(m):(sv[:,m:].square().sum(-1)/energy).tolist() for m in [1,8,16,32,64,128,256]},
            ranks={str(f):((sv.square().cumsum(-1)/energy[:,None]<f).sum(-1)+1).tolist() for f in [.9,.99,.999]}))
    log_energy=torch.tensor([r['log_energy'] for r in spectra],dtype=torch.float64);weights=log_energy.softmax(0)
    results['svd']=dict(documents=spectra,
        pooled_relative_floor={str(m):(weights*torch.tensor([r['residuals'][str(m)] for r in spectra],dtype=torch.float64)).sum(0).tolist()
                               for m in [1,8,16,32,64,128,256]},
        mean_document_relative_floor={str(m):torch.tensor([r['residuals'][str(m)] for r in spectra],dtype=torch.float64).mean(0).tolist()
                                      for m in [1,8,16,32,64,128,256]},
        note='Signed rank-m oracle for each held-out 512x512 legal rectangular block. Not a population bound or realizable positive feature map.')
    (P/'distribution.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({k:(v if k!='svd' else {a:b for a,b in v.items() if a!='documents'}) for k,v in results.items()},indent=2))


if __name__=='__main__':main()
