"""Paired document bootstrap, fixed-checkpoint summaries, protocol audit."""
import json, hashlib, math, platform, subprocess
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
KINDS={'gla':['ad_shape','ad_full','exp_mlp','signed_residual'],
       'gdn':['ad_norm','ad_write','exp_mlp','signed_residual']}
def main():
 table=[];paired=[];timing=[];audit=[]
 manifest=read(P/'data/manifest.json');recs=manifest['records']
 assert len(recs)==1136
 for field in ['text_sha256','token_sha256']:
  assert len({r[field] for r in recs})==len(recs)
 for r in recs:
  assert hashlib.sha256(json.dumps(r['input_ids']).encode()).hexdigest()==r['token_sha256']
  assert len(r['input_ids'])==(2048 if r['split']=='confirm_long' else 512)
 for f,kinds in KINDS.items():
  mainplan=read(P/'results'/f'{f}_plan.json');followplan=read(P/'results'/f'{f}_followup_plan.json')
  frozen=read(P/'checks'/f'{f}_frozen.json');loading=read(P/'checks'/f'{f}_load.json')
  assert mainplan['base_sha256']==followplan['base_sha256']==frozen['base_sha256'] and frozen['passed']
  assert not any(loading['loading'].values())
  orders={};counts=set();first_gradients=True
  for k in kinds:
   for seed in [11,29]:
    fit=read(P/'fits'/f'{f}_{k}_s{seed}.json')
    assert fit['epochs']==1 and fit['steps']==256 and fit['training_docs']==1024
    assert fit['training_tokens']==524288 and fit['prediction_targets']==523264
    if seed in orders:assert orders[seed]==fit['order_sha256']
    orders[seed]=fit['order_sha256'];counts.add(fit['params'])
    first_gradients &= all(v is not None and math.isfinite(v) and v>0 for v in fit['first_gradient_norms'].values())
  assert len(counts)==1 and first_gradients
  audit.append(dict(family=f,fits=8,matched_registered_parameters=counts.pop(),base_frozen=True,
   all_single_pass=True,orders_match_by_seed=True,nonzero_first_matrix_gradients=first_gradients,
   loading_keys_clean=True,effective_capacity_caveat='Amplitude outputs cancel or are unused in some ablations; counts are registered parameters.'))
  for split in ['validation','confirm_wiki','confirm_long']:
   arrays={};tokens=None
   for k in ['native']+kinds:
    names=['native'] if k=='native' else [f'{f}_{k}_s{s}' for s in [11,29]]
    results=[read(P/'results'/f'ppl_{f}_{split}_{name}.json') for name in names]
    indices=[d['index'] for d in results[0]['documents']]
    for result in results:assert [d['index'] for d in result['documents']]==indices
    tok=np.array([d['tokens'] for d in results[0]['documents']])
    if tokens is not None:assert np.array_equal(tok,tokens)
    tokens=tok
    arrays[k]=np.mean([[d['nll'] for d in result['documents']] for result in results],axis=0)
    ppls=[r['ppl'] for r in results]
    table.append(dict(family=f,split=split,kind=k,ppl_mean=float(np.mean(ppls)),
     ppl_by_seed=ppls,ppl_seed_std=float(np.std(ppls,ddof=1)) if len(ppls)>1 else 0.,
     nll_seed_mean=float(arrays[k].sum()/tokens.sum()),documents=len(indices)))
   ad='ad_full' if f=='gla' else 'ad_write'
   direction='ad_shape' if f=='gla' else 'ad_norm'
   for other in ['native',direction,'exp_mlp','signed_residual']:
    delta=arrays[ad]-arrays[other]
    rng=np.random.default_rng(20260908)
    ix=rng.integers(0,len(tokens),(20000,len(tokens)))
    boots=delta[ix].sum(-1)/tokens[ix].sum(-1)
    low,high=np.quantile(boots,[.025,.975]);point=float(delta.sum()/tokens.sum())
    paired.append(dict(family=f,split=split,comparison=f'{ad} minus {other}',
     nll_delta=point,document_bootstrap_95ci=[float(low),float(high)],
     relative_ppl_change_pct=100*math.expm1(point),
     relative_ppl_change_95ci_pct=[100*math.expm1(float(low)),100*math.expm1(float(high))],
     document_win_fraction=float(np.mean(delta<0))))
  for k in ['native']+kinds:
   name='native' if k=='native' else f'{f}_{k}_s11'
   path=P/'results'/f'timing_{f}_{name}.json'
   if path.exists():timing.append(dict(kind=k,**read(path)))
 result=dict(ppl=table,paired=paired,timing=timing,audit=audit,
  data=dict(counts=manifest['counts'],cross_split_document_and_token_hashes_unique=True,
   manifest_sha256=hashlib.sha256((P/'data/manifest.json').read_bytes()).hexdigest()),
  confidence_scope='20,000 paired document resamples of seed-mean NLL, conditional on fitted checkpoints and this document set. Not a training-seed uncertainty interval, population bound, or blind confirmatory test.')
 save(P/'results/summary.json',result)
 for row in table:
  if row['split']!='validation':print(row['family'],row['split'],row['kind'],round(row['ppl_mean'],5))
 for row in paired:
  if row['split']!='validation':print('paired',row)
 if timing:
  for row in timing:print('time',row['family'],row['kind'],row['prefill_512_ms'],row['decode_ms_per_token'],row['state_bytes'])
if __name__=='__main__':main()
