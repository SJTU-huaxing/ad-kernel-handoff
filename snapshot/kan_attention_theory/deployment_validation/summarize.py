"""Paired document uncertainty, implementation checks, and compact result tables."""
import csv,json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent

def main():
    primary=P/'results/ppl_optimized.json'
    if not primary.exists():primary=P/'results/ppl.json'
    rows=json.loads(primary.read_text())['results'];rng=np.random.default_rng(20260920)
    comparisons=[];summary=[]
    for ds in ['internal','official']:
        group=[r for r in rows if r['dataset']==ds]
        teacher=next(r for r in group if r['method']=='teacher')
        partition=next(r for r in group if r['method']=='partition')
        base=np.array([d['nll'] for d in teacher['documents']]);part=np.array([d['nll'] for d in partition['documents']])
        ix=rng.integers(0,len(base),(10000,len(base)))
        for r in group:
            values=np.array([d['nll'] for d in r['documents']]);delta=values-base
            ratio=np.exp(delta[ix].mean(1))-1
            summary.append(dict(dataset=ds,method=r['method'],seed=r['seed'],perplexity=r['perplexity'],
                relative_ppl_change=r['perplexity']/teacher['perplexity']-1,
                paired_bootstrap_relative_change_95=np.quantile(ratio,[.025,.975]).tolist()))
            if r['method'] in ['favor_plus','favor_plus_640']:
                diff=part-values;boot=diff[ix].mean(1)
                comparisons.append(dict(dataset=ds,comparison='partition_minus_favor_nll',comparator=r['method'],seed=r['seed'],
                    mean=float(diff.mean()),paired_document_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),
                    documents_partition_better=int((diff<0).sum()),documents=len(diff)))
    f64=json.loads((P/'results/ppl_fp64.json').read_text())['results'];precision=[]
    for r in f64:
        r32=next(x for x in rows if all(x[k]==r[k] for k in ['method','seed','dataset']))
        a=np.array([d['nll'] for d in r['documents']]);b=np.array([d['nll'] for d in r32['documents'][:4]])
        precision.append(dict(method=r['method'],seed=r['seed'],dataset=r['dataset'],
            max_document_nll_difference=float(abs(a-b).max()),mean_nll_difference=float((a-b).mean())))
    package=dict(ppl=summary,paired_comparisons=comparisons,precision_checks=precision,
        bootstrap='10000 paired document resamples, percentile 95% intervals. Approximate sample uncertainty, not certified population bounds. No token/pair independence assumption.',
        scope='4/336 query heads, m64, fixed 1024-token document prefixes, no model finetuning.')
    (P/'results/summary.json').write_text(json.dumps(package,indent=2))
    with (P/'ppl.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print(json.dumps(package,indent=2))

if __name__=='__main__':main()
