"""End-to-end chunk vs cached recurrence, with actual heldout continuations."""
import native as n
from native import *
from assess import install, state_bytes

@torch.no_grad()
def main(family):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 model=model_load(family);norms=torch.load(P/'data'/f'{family}_norm.pt',weights_only=True)
 docs=[r for r in data_load()['records'] if r['split']=='confirm_wiki'][:2]
 names=['native']+[f'{family}_{k}_s11' for k in KINDS[family]+['signed_residual']]
 rows=[]
 for name in names:
  maps=install(model,family,name,norms)
  for doc in docs:
   x=torch.tensor([doc['input_ids'][:160]],device='cuda')
   full=model(x,use_cache=False).logits[:,127:159].float()
   out=model(x[:,:128],use_cache=True);cache=out.past_key_values
   start_bytes=state_bytes(cache);pieces=[out.logits[:,-1:].float()]
   for j in range(128,159):
    out=model(x[:,j:j+1],past_key_values=cache,use_cache=True)
    cache=out.past_key_values;pieces.append(out.logits.float())
   cached=torch.cat(pieces,dim=1)
   logp=full.log_softmax(-1);logc=cached.log_softmax(-1)
   kl=(logp.exp()*(logp-logc)).sum(-1).mean()
   rel=(cached-full).norm()/full.norm()
   nll_full=F.cross_entropy(full.flatten(0,1),x[:,128:160].flatten())
   nll_cache=F.cross_entropy(cached.flatten(0,1),x[:,128:160].flatten())
   row=dict(name=name,document=doc['index'],logit_relative_l2=float(rel),
    mean_teacher_kl=float(kl),nll_delta=float(nll_cache-nll_full),
    prefix128_state_bytes=start_bytes,prefix159_state_bytes=state_bytes(cache),
    passed=bool(torch.isfinite(cached).all() and rel<.035 and kl<.01 and start_bytes==state_bytes(cache)))
   rows.append(row);print(json.dumps(row),flush=True)
  n.MAPS=None
 result=dict(family=family,rows=rows,passed=all(r['passed'] for r in rows),
  scope='Two heldout documents, 128-token prefill and 31 cached steps; compare 32 next-token logits to length160 chunk forward. BF16 tolerance and fixed-state bytes.')
 save(P/'checks'/f'{family}_cache.json',result)
 assert result['passed']

if __name__=='__main__':main(sys.argv[1])
