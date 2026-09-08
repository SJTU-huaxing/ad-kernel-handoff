"""Heldout key geometry and controlled associative recall; not LM benchmarks."""
import native as n
from native import *
from theory_checks import delta,rms
from assess import install

@torch.no_grad()
def collect(model,family):
 path=P/'data'/f'{family}_heldout_operators.pt'
 if path.exists():return torch.load(path,weights_only=True)
 n.MAPS=None;rows=[]
 docs=[r for r in data_load()['records'] if r['split']=='confirm_wiki'][:8]
 for r in docs:
  n.CAPTURE={};model(torch.tensor([r['input_ids']],device='cuda'),use_cache=False,logits_to_keep=1)
  rows.append(n.CAPTURE)
 n.CAPTURE=None;torch.save(rows,path);return rows

@torch.no_grad()
def main(family):
 torch.set_num_threads(4);model=model_load(family);norms=torch.load(P/'data'/f'{family}_norm.pt',weights_only=True)
 data=collect(model,family);rows=[]
 names=['native']+[f'{family}_{kind}_s11' for kind in KINDS[family]]
 if (P/'fits'/f'{family}_signed_residual_s11.pt').exists():names.append(f'{family}_signed_residual_s11')
 for name in names:
  maps=install(model,family,name,norms)
  for l in FAMILY_LAYERS[family]:
   for di,record in enumerate(data):
    rr=record[l];q,k,v=[rr[s][:,::4].cuda() for s in ['q','k','v']]
    if maps is None:kf=k.float();amp=None
    elif maps[str(l)].kind=='signed_residual':
     _,kf,_=maps[str(l)](q,k,v);kf=kf.float();amp=None
    else:
     kf,amp=maps[str(l)].feature(maps[str(l)].coordinates(k,'k'))
    u=F.normalize(kf[0].permute(1,0,2).double(),dim=-1)
    gram=u@u.transpose(-1,-2);off=~torch.eye(u.shape[1],device='cuda',dtype=torch.bool)
    # Same unit keys for read/write, random independent values. Isolates key geometry.
    g=torch.Generator(device='cuda').manual_seed(927000+di+l)
    val=torch.randn(u.shape[0],u.shape[1],32,generator=g,device='cuda',dtype=torch.float64)
    ones=torch.ones(u.shape[:2],device='cuda',dtype=torch.float64)
    _,_,state=delta(u,u,val,ones,ones)
    read=u@state
    nmse=(read-val).square().sum((-1,-2))/val.square().sum((-1,-2))
    row=dict(name=name,layer=l,document=di,mean_key_cosine=float(gram[:,off].mean()),
     p95_key_cosine=float(gram[:,off].quantile(.95)),
     associative_recall_nmse=float(nmse.mean()),
     scope='128 heldout native keys; synthetic independent values; alpha=beta=1; key used for both writing and reading. Diagnoses geometry, not native-gate PPL.')
    if amp is not None:row.update(amplitude_mean=float(amp.mean()),amplitude_min=float(amp.min()),amplitude_max=float(amp.max()))
    rows.append(row)
  n.MAPS=None
 save(P/'results'/f'{family}_geometry.json',dict(rows=rows))
 print(family,'geometry complete',flush=True)

if __name__=='__main__':main(sys.argv[1])
