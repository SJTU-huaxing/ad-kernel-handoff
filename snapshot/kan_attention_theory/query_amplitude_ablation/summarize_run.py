import hashlib
import itertools
import numpy as np
from ablation import *


def read(path): return json.loads(path.read_text())


def groups(regime):
    result = {}
    for variant in VARIANTS:
        lr = plan()['learning_rates'][f'{regime}__{variant}']
        result[variant] = [(P,f'{regime}_{variant}_s{seed}_lr{lr:g}') for seed in SEEDS]
    for kind in ['ad','hh'] + (['hh_softmax'] if regime == 'causal_kl' else []):
        result[kind] = [(OLD,old_name(regime,kind,seed)) for seed in SEEDS]
    return result


def fetch(items, prefix):
    files = [root/'results'/f'{prefix}_{name}.json' for root,name in items]
    return [read(f) for f in files] if all(f.exists() for f in files) else None


def paired_docs(a,b):
    delta = np.asarray(a)-np.asarray(b)
    means = delta.mean(0)
    rng = np.random.default_rng(986351)
    ix = rng.integers(0,len(means),size=(5000,len(means)))
    return dict(mean_difference=float(delta.mean()),seed_differences=delta.mean(-1).tolist(),
                paired_document_95_ci=np.quantile(means[ix].mean(-1),[.025,.975]).tolist(),
                scope='Paired document bootstrap conditional on three fixed fits; no correction for exploratory multiple comparisons.')


def main():
    kernels,products,ppls,comparisons,product_comparisons = [],[],[],[],[]
    for regime in REGIMES:
        gg = groups(regime)
        for group,items in gg.items():
            for split in ['confirm_wiki','confirm_long']:
                values = fetch(items,f'kernel_{split}')
                if values:
                    row = dict(regime=regime,group=group,split=split,
                               parameters_per_head=values[0]['metadata']['parameters_per_head'],m=values[0]['metadata']['m'])
                    for key in ['relative_raw_i','raw_nmse','balanced','kl','mass','output_nmse','log_mass_mae']:
                        a = np.asarray([v['summary'][key] for v in values])
                        row[key],row[key+'_seed_means'] = float(a.mean()),a.mean(-1).tolist()
                        row[key+'_head_median'] = float(np.median(a.mean(0)))
                    kernels.append(row)
                values = fetch(items,f'ppl_{split}')
                if values:
                    ppls.append(dict(regime=regime,group=group,split=split,ppl=float(np.mean([v['ppl'] for v in values])),
                                     seed_ppls=[v['ppl'] for v in values]))
            for direction in ['ab','ba']:
                values = fetch(items,f'product_{direction}')
                if values:
                    row = dict(regime=regime,group=group,direction=direction)
                    for key in ['relative_raw_i','raw_nmse','balanced','kl','mass']:
                        a = np.asarray([v['summary'][key] for v in values])
                        row[key],row[key+'_seed_means'] = float(a.mean()),a.mean(-1).tolist()
                        row[key+'_head_median'] = float(np.median(a.mean(0)))
                    products.append(row)
        pairs = [('reduced_matched','ad'),('reduced_matched','hh'),('reduced_plain','ad'),('reduced_matched','reduced_plain')]
        if regime == 'causal_kl':pairs.append(('reduced_matched','hh_softmax'))
        for ga,gb in pairs:
            for split in ['confirm_wiki','confirm_long']:
                for metric in ['balanced','kl','output_nmse','nll']:
                    prefix = f'{"ppl" if metric=="nll" else "kernel"}_{split}'
                    va,vb = fetch(gg[ga],prefix),fetch(gg[gb],prefix)
                    if va is None or vb is None:continue
                    def extract(vv):
                        return [[d['nll']/d['tokens'] if metric=='nll' else float(np.mean(d[metric]))
                                 for d in v['documents']] for v in vv]
                    comparisons.append(dict(regime=regime,a=ga,b=gb,split=split,metric=metric,
                                            **paired_docs(extract(va),extract(vb))))
            for direction in ['ab','ba']:
                va,vb = fetch(gg[ga],f'product_{direction}'),fetch(gg[gb],f'product_{direction}')
                if va is None or vb is None:continue
                counts = np.random.default_rng(86523).multinomial(64,np.full(64,1/64),size=5000)
                for num,den,metric in [('raw_i','raw_truth_mass','relative_raw_i'),('raw_sse','raw_energy','raw_nmse')]:
                    def bootstrap(vv):
                        n = np.asarray([v['per_query_document'][num] for v in vv]).mean(0)
                        d = np.asarray([v['per_query_document'][den] for v in vv]).mean(0)
                        return ((counts@n.T)/(counts@d.T)).mean(-1)
                    ah = np.asarray([v['summary'][metric] for v in va]).mean(0)
                    bh = np.asarray([v['summary'][metric] for v in vb]).mean(0)
                    has_documents = all('per_query_document' in v for v in va+vb)
                    if regime == 'product_i':
                        assert has_documents, (ga, gb, direction)
                    delta = bootstrap(va)-bootstrap(vb) if has_documents else None
                    product_comparisons.append(dict(regime=regime,a=ga,b=gb,direction=direction,metric=metric,
                         mean_difference=float((ah-bh).mean()),a_better_heads=int((ah<bh).sum()),total_heads=24,
                         paired_query_document_95_ci=np.quantile(delta,[.025,.975]).tolist() if has_documents else None,
                         scope='Conditional on fixed heldout K bank and three fits; Q-document resampling only.' if has_documents else 'Parent causal-fit product files lack document sufficient statistics; point estimates only.'))
    fits = [read(f) for f in sorted((P/'fits').glob('*.json'))]
    save(P/'results'/'summary.json',dict(kernel=kernels,product=products,ppl=ppls,comparisons=comparisons,
         product_comparisons=product_comparisons,new_fit_count=len(fits),selected_fit_count=len(plan()['models']),
         total_training_seconds=sum(v['metadata']['training_seconds'] for v in fits),
         scope='New models evaluated on previously used heldout documents; parent AD/HH results reused from identical frozen protocols.'))
    print(json.dumps(dict(kernel=kernels,product=products,ppl=ppls),indent=2))


if __name__=='__main__':main()
