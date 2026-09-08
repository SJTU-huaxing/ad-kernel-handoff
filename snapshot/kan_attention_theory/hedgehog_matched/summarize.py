"""Read-only aggregation of experimental outputs, paired document intervals."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parent
R = P / 'results'


def read(path):
    return json.loads(path.read_text())


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))


def ci(a, b):
    delta = np.asarray(a) - np.asarray(b)
    mean_docs = delta.mean(0)
    rng = np.random.default_rng(986351)
    ix = rng.integers(0, len(mean_docs), size=(5000, len(mean_docs)))
    return dict(mean=float(delta.mean()), per_seed_mean=delta.mean(-1).tolist(),
                paired_document_95_ci=np.quantile(mean_docs[ix].mean(-1), [.025, .975]).tolist(),
                scope='Conditional on these three fitted seeds; paired documents, '
                      'not independent head/token or full training-randomness CI.')


def main():
    product_stage = os.environ.get('MATCHED_PLAN') == 'product_plan.json'
    plan = read(R / ('product_plan.json' if product_stage else 'frozen_plan.json'))
    def kind_names(kind):
        return [n for n in plan['models']
                if read(P/'fits'/f'{n}.json')['metadata']['kind'] == kind]
    rows, products, ppls, comparisons = [], [], [], []
    for kind in plan['kinds']:
        names = kind_names(kind)
        for split in ['confirm_wiki', 'confirm_long']:
            files = [R / f'kernel_{split}_{name}.json' for name in names]
            if all(f.exists() for f in files):
                vals = [read(f) for f in files]
                row = dict(kind=kind, split=split, m=vals[0]['metadata']['m'],
                           parameters_per_head=vals[0]['metadata']['parameters_per_head'])
                for key in ['relative_raw_i', 'raw_nmse', 'balanced', 'kl',
                            'mass', 'output_nmse', 'log_mass_mae']:
                    a = np.asarray([v['summary'][key] for v in vals])
                    row[key] = float(a.mean())
                    row[key+'_seed_means'] = a.mean(-1).tolist()
                    row[key+'_median_head'] = float(np.median(a.mean(0)))
                rows.append(row)
            files = [R/f'ppl_{split}_{name}.json' for name in names]
            if all(f.exists() for f in files):
                vals = [read(f)['ppl'] for f in files]
                ppls.append(dict(kind=kind, split=split, ppl=float(np.mean(vals)),
                                 seed_ppls=vals))
        for direction in ['ab', 'ba']:
            files = [R/f'product_{direction}_{name}.json' for name in names]
            if all(f.exists() for f in files):
                vals = [read(f) for f in files]
                row = dict(kind=kind, direction=direction)
                for key in ['relative_raw_i', 'raw_nmse', 'balanced', 'kl', 'mass']:
                    a = np.asarray([v['summary'][key] for v in vals])
                    row[key] = float(a.mean())
                    row[key+'_seed_means'] = a.mean(-1).tolist()
                    row[key+'_median_head'] = float(np.median(a.mean(0)))
                products.append(row)

    for split in ['confirm_wiki', 'confirm_long']:
        for ka, kb in [('ad_raw', 'hh_raw'), ('ad_kl', 'hh_kl'),
                       ('ad_kl', 'hh_softmax_kl'), ('ad_raw', 'hh_kl'),
                       ('ad_raw', 'hh_softmax_kl')]:
            if ka not in plan['kinds'] or kb not in plan['kinds']:
                continue
            def names(kind):
                return kind_names(kind)
            for metric in ['balanced', 'kl', 'output_nmse', 'nll']:
                def vals(kind):
                    files = [R/f'{"ppl" if metric=="nll" else "kernel"}_{split}_{n}.json'
                             for n in names(kind)]
                    if not all(f.exists() for f in files):
                        return None
                    return [[r['nll']/r['tokens'] if metric == 'nll' else float(np.mean(r[metric]))
                             for r in read(f)['documents']] for f in files]
                a, b = vals(ka), vals(kb)
                if a is not None and b is not None:
                    comparisons.append(dict(split=split, a=ka, b=kb, metric=metric, **ci(a,b)))
    fits = [read(f) for f in sorted((P/'fits').glob('*.json'))
            if f.name.startswith('product_') == product_stage]
    product_comparisons=[]
    if product_stage:
        for direction in ['ab','ba']:
            paths={kind:[R/f'product_{direction}_{n}.json' for n in kind_names(kind)]
                   for kind in ['ad_raw','hh_raw']}
            if all(f.exists() for fs in paths.values() for f in fs):
                values={kind:[read(f) for f in fs] for kind,fs in paths.items()}
                rng=np.random.default_rng(86523)
                counts=rng.multinomial(64,np.full(64,1/64),size=5000)
                for numerator,denominator,key in [('raw_i','raw_truth_mass','relative_raw_i'),
                                                  ('raw_sse','raw_energy','raw_nmse')]:
                    head={kind:np.asarray([v['summary'][key] for v in vv]).mean(0)
                          for kind,vv in values.items()}
                    boots={}
                    for kind,vv in values.items():
                        num=np.asarray([v['per_query_document'][numerator] for v in vv]).mean(0)
                        den=np.asarray([v['per_query_document'][denominator] for v in vv]).mean(0)
                        boots[kind]=((counts@num.T)/(counts@den.T)).mean(-1)
                    delta=boots['ad_raw']-boots['hh_raw']
                    product_comparisons.append(dict(direction=direction,metric=key,
                        mean_difference=float((head['ad_raw']-head['hh_raw']).mean()),
                        ad_better_heads=int((head['ad_raw']<head['hh_raw']).sum()),
                        total_heads=24,paired_query_document_95_ci=np.quantile(delta,[.025,.975]).tolist(),
                        scope='Query-document bootstrap conditional on fixed heldout K bank and three fits; '
                              'does not cover unknown P_K or full training randomness.'))
    out = dict(kernel=rows, product=products, ppl=ppls, comparisons=comparisons,
               product_comparisons=product_comparisons,
               fit_count=len(fits),
               total_training_seconds=sum(f['metadata']['training_seconds'] for f in fits),
               scope='Paired observations from fixed, previously used heldout documents; '
                     'no claim of population-optimality or full published training-system replication.')
    save(R/('summary_product.json' if product_stage else 'summary.json'), out)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
