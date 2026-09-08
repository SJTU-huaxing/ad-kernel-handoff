"""Final evidence completeness, budget/data invariants and artifact provenance."""
import json,hashlib,sys,subprocess
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def main():
 fits=list((P/'fits').glob('*.pt'));assert len(fits)==36,len(fits)
 kinds=['split_1','split_01','split_kl','exp_kl','softplus_kl','hedgehog','learned_prf'];maink=kinds[:5];metas={f.stem:torch.load(f,weights_only=True,map_location='cpu')['metadata'] for f in fits};counts={}
 for seed in [11,29,47]:
  cohort=[metas[f'{k}_s{seed}'] for k in kinds];assert len({r['query_sha256'] for r in cohort})==len({r['order_sha256'] for r in cohort})==len({r['training_pairs_per_head'] for r in cohort})==1
  for k in maink:
   m=metas[f'{k}_s{seed}'];box=torch.load(P/'fits'/f'{k}_s{seed}.pt',weights_only=True,map_location='cpu');n=sum(v.numel() for key,v in box['state_dict'].items() if key.startswith(('qnet.','knet.')))//24;assert n==m['parameters_per_head']==73856
  for k in ['calibrated','gate']:
   m=metas[f'{k}_s{seed}'];assert m['calibration_pairs_disjoint'] and m['calibration_steps']==4096 and m['parameters_per_head']==73856
 manifest=read(P/'data/manifest.json');splits={s:[r for r in manifest['records'] if r['split']==s] for s in ['train','validation','confirm_wiki','confirm_long','confirm_prose']}
 for s,rows in splits.items():counts[s]=len(rows)
 assert counts==dict(train=4096,validation=128,confirm_wiki=128,confirm_long=24,confirm_prose=128)
 for typ in ['text_sha256','token_sha256']:
  for s in ['confirm_wiki','confirm_long','confirm_prose']:
   for t in splits:
    if s!=t:assert not {r[typ] for r in splits[s]}&{r[typ] for r in splits[t]},(s,t,typ)
 seen=0
 for i,f in enumerate(sorted((P/'data').glob('calibration_*.pt'))):
  b=torch.load(f,weights_only=True,map_location='cpu')
  for j,pos in enumerate(b['query_positions']):
   original=set(splits['train'][i*32+j]['query_positions']);assert len(pos.unique())==64 and not set(pos.tolist())&original;seen+=1
 assert seen==4096
 kernels=list((P/'results').glob('kernel_*.json'));ppl=list((P/'results').glob('ppl_*.json'));assert len(kernels)==108,len(kernels);assert len(ppl)==132,len(ppl)
 for f in kernels+ppl:
  a=read(f);assert len(a['documents'])==counts[a['split']]
 for name in ['operators','model_cache','long_precision_and_window','theory']:
  assert read(P/'checks'/f'{name}.json')['passed']
 for s in ['confirm_wiki','confirm_long','confirm_prose']:
  a=read(P/'results'/f'witness_{s}.json');assert len(a['rows'])==24;assert all(r['curves'][2]['balanced_iid_document_95_lower_bound']==0 for r in a['rows'])
 causal=read(P/'results/causal_witness.json');assert len(causal['documents'])==128
 bench=read(P/'results/benchmark.json');assert len(bench['results'])==24
 for r in bench['results']:
  n=r['prompt_tokens'];expected=28*1024*n if r['name']=='teacher' else 26*1024*n;assert r['kv_bytes']==expected,(r['name'],r['kv_bytes'],expected)
  if r['name']!='teacher':assert r['state_bytes']==(0 if r['mode']=='window' else 798720)
 artifacts=[]
 for folder,pattern in [('fits','*.pt'),('results','*.json'),('checks','*.json')]:
  for f in sorted((P/folder).glob(pattern)):
   if f.name in ['artifact_manifest.json','final_audit.json']:continue
   artifacts.append(dict(path=str(f.relative_to(P)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
 for f in sorted(P.glob('*.py')):artifacts.append(dict(path=f.name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
 (P/'results/artifact_manifest.json').write_text(json.dumps(artifacts,indent=2));result=dict(passed=True,fits=len(fits),matched_main_fits=15,published_form_fits=6,calibrations=12,favor_banks=3,kernel_evaluations=len(kernels),full_model_ppl_evaluations=len(ppl),benchmark_cases=24,data_counts=counts,training_pairs_per_head=metas['split_01_s11']['training_pairs_per_head'],parameters_per_head=73856,calibration_pairs_disjoint=True,model_revision=manifest['revision'],scope='Two full GQA layers,24/336heads; no LLM finetuning; all candidate results retained; finite empirical and population claims separated.');(P/'checks/final_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
