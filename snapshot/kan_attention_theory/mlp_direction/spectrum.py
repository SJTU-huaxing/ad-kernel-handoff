"""Conditional-context spectral proxies; these are diagnostics, not a population certificate."""
import itertools
from core import *

@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    manifest=json.loads((DATA/'manifest.json').read_text())
    chosen=torch.randperm(4096,generator=torch.Generator().manual_seed(73401))[:64].tolist()
    by={}
    for i in chosen:by.setdefault(i//64,[]).append(i%64)
    ranks=[16,32,64,96,128];out=[]
    for shard,offsets in sorted(by.items()):
        b=torch.load(DATA/f'train_{shard:03d}.pt',weights_only=True)
        for off in offsets:
            q=b['q'][off,:,::2].cuda().double();k=b['k'][off,:,::2].cuda().double()
            logk=q@k.transpose(-1,-2)/math.sqrt(128)
            logz=logk.logsumexp(-1,keepdim=True)
            curves={}
            for beta,name in [(0.,'raw'),(.5,'sqrt_mass'),(1.,'row_mass')]:
                logits=logk-beta*logz
                logits=logits-logits.amax((-1,-2),keepdim=True)
                mat=logits.exp();s=torch.linalg.svdvals(mat);energy=s.square()
                curves[name]=torch.stack([energy[:,m:].sum(-1)/energy.sum(-1) for m in ranks],-1).tolist()
            out.append(dict(ordinal=shard*64+off,curves=curves))
    means={name:torch.tensor([r['curves'][name] for r in out],dtype=torch.float64).mean(0).tolist() for name in ['raw','sqrt_mass','row_mass']}
    allocations={}
    # Exactly sum m=256, making the sum of hidden/output parameter counts equal.
    configs=[ms for ms in itertools.product(ranks,repeat=4) if sum(ms)==256]
    for name,table in means.items():
        best=min(configs,key=lambda ms:sum(table[h][ranks.index(m)] for h,m in enumerate(ms)))
        allocations[name]=list(best)
    save(P/'results/spectrum.json',dict(ranks=ranks,documents=out,mean_relative_tails=means,allocations=allocations,
        protocol='64 fixed random training documents, 256x256 per head. Mean document-relative spectral tails. Conditional empirical operators; no test fitting; not a q-only deployed row normalization or population bound.'))
    print(json.dumps(dict(event='spectra',means=means,allocations=allocations)),flush=True)

if __name__=='__main__':main()
