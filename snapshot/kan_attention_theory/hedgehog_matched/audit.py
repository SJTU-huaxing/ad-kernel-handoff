import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
plan = read(P/'results/frozen_plan.json')
selection = read(P/'results/selection.json')
assert hashlib.sha256((P/'PROTOCOL.zh.md').read_bytes()).hexdigest() == selection['protocol_sha256']
fits = [f for f in (P/'fits').glob('*.json') if not f.name.startswith('product_')]
assert len(fits) == 20
by_seed = {}
for name in plan['models']:
    row = read(P/'fits'/f'{name}.json')
    meta = row['metadata']
    assert meta['train_documents'] == 4096 and meta['epochs'] == 1
    assert meta['training_queries_per_head'] == 4096*64
    assert meta['parameters_per_head'] == (73729 if meta['kind'].endswith('_raw') else 73728)
    assert meta['m'] == (64 if meta['kind'].startswith('ad') else 576)
    assert meta['lr'] == selection['learning_rates'][meta['kind']]
    by_seed.setdefault(meta['seed'], set()).add((meta['order_sha256'], meta['query_sha256'],
                                               meta['training_pairs_per_head']))
    for split in ['confirm_wiki', 'confirm_long']:
        for prefix in ['kernel', 'ppl']:
            assert (P/'results'/f'{prefix}_{split}_{name}.json').exists()
    for direction in ['ab', 'ba']:
        assert (P/'results'/f'product_{direction}_{name}.json').exists()
assert all(len(v) == 1 for v in by_seed.values())
assert read(P/'checks/operators.json')['passed']
assert read(P/'checks/long_precision.json')['passed']
assert len(read(P/'results/benchmark.json')['rows']) == 6
summary = read(P/'results/summary.json')
assert len(summary['kernel']) == 10 and len(summary['ppl']) == 10 and len(summary['product']) == 10
for split, expected in [('confirm_wiki', 8.664559867077683), ('confirm_long', 9.066279)]:
    actual = read(P/'results'/f'ppl_{split}_teacher.json')['ppl']
    assert abs(actual-expected) < 1e-4, (split, actual, expected)
result = dict(passed=True, fits=20, selected_fits=15, kernel_evaluations=30,
              empirical_product_evaluations=30, ppl_evaluations=32,
              timing_conditions=6, same_pairs_and_order_within_seed=True,
              exact_trainable_parameter_matching=True,
              parameter_scope='Raw-I main pairs73729/head. KL controls73728 registered trainable '
                              'parameters, but AD query-amplitude192 weights are not identified by '
                              'direction KL; not a match of statistically identifiable degrees of freedom.',
              scope='No full-model weights trained; existing heldout sets; no population bound.')
if (P/'results/product_plan.json').exists():
    product_plan=read(P/'results/product_plan.json')
    product_selection=read(P/'results/product_selection.json')
    assert hashlib.sha256((P/'PRODUCT_PROTOCOL.zh.md').read_bytes()).hexdigest()==product_selection['protocol_sha256']
    assert len(list((P/'fits').glob('product_*.json')))==8
    for name in product_plan['models']:
        meta=read(P/'fits'/f'{name}.json')['metadata']
        assert meta['parameters_per_head']==73729 and meta['epochs']==1
        assert meta['training_pairs_per_head']==4096*64*1024
        for split in ['confirm_wiki','confirm_long']:
            for prefix in ['kernel','ppl']:
                assert (P/'results'/f'{prefix}_{split}_{name}.json').exists()
        for direction in ['ab','ba']:
            assert (P/'results'/f'product_{direction}_{name}.json').exists()
    assert read(P/'checks/operators_product.json')['passed']
    assert read(P/'checks/long_precision_product.json')['passed']
    extra=read(P/'results/summary_product.json')
    assert len(extra['kernel'])==4 and len(extra['ppl'])==4 and len(extra['product'])==4
    result['product_followup']=dict(fits=8,selected_fits=6,kernel_evaluations=12,
                                    empirical_product_evaluations=12,ppl_evaluations=12,
                                    scope='Post-first-stage mechanism experiment, explicitly disclosed.')
(P/'checks/final_audit.json').write_text(json.dumps(result, indent=2))
paths = [p for p in P.rglob('*') if p.is_file() and p.suffix in ['.py','.md','.json','.pt']
         and p.name != 'artifact_manifest.json']
manifest = {str(p.relative_to(P)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(P/'results/artifact_manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(result, indent=2))
