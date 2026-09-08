import hashlib
import numpy as np
from common_eval import *

def main():
    s=read(P/'results/summary.json')
    assert len(s['table'])==8 and len(s['diagnostics'])==8 and len(s['tail'])==4
    assert len(s['comparisons'])==36
    for stem in ['operators','long_precision']:
        assert read(P/'checks'/f'{stem}.json')['passed']
    errors=[]
    for split,ndocs in [('confirm_wiki',128),('confirm_long',24)]:
        d=read(P/'results'/f'diagnostics_{split}.json')
        assert len(d['models'])==12
        assert d['max_discrepancy_with_parent_metrics']<1e-9
        errors.append(d['max_discrepancy_with_parent_metrics'])
        for rows in d['models'].values():
            assert len(rows)==ndocs
        for g in GROUPS:
            for seed in SEEDS:
                k=read(parent_file(g,seed,split,'kernel'));p=read(parent_file(g,seed,split,'ppl'))
                assert len(k['documents'])==ndocs and len(p['documents'])==ndocs
                assert all(d['tokens']==(1023 if split=='confirm_wiki' else 8191) for d in p['documents'])
        teacher=read(P/'results'/f'ppl_{split}_teacher.json')
        original=read(ROOT/'query_amplitude_ablation/results'/f'ppl_{split}_teacher.json')
        assert teacher['ppl']==original['ppl']
    assert all(r['difference']==0 for r in s['favor_recheck'])
    for r in s['tail']:
        expected=next(v['kl'] for v in s['table'] if v['split']=='confirm_long' and v['group']==r['group'])
        assert abs(r['positive_kl']+r['negative_kl']-expected)<1e-10
    # Exact illustrative non-monotonicity, not a model performance test.
    p=np.asarray([.5,.25,.25]);a=np.asarray([.5,.49,.01]);b=np.asarray([.45,.275,.275]);v=np.asarray([0.,1.,1.])
    toy=dict(kl_a=float(np.sum(p*np.log(p/a))),kl_b=float(np.sum(p*np.log(p/b))),
        output_error_a=float(abs((a-p)@v)),output_error_b=float(abs((b-p)@v)))
    assert toy['kl_a']>toy['kl_b'] and toy['output_error_a']<toy['output_error_b']
    save(P/'checks/final_audit.json',dict(passed=True,max_recomputed_metric_errors=errors,
        favor_ppl_recheck_exact=True,toy=toy,scope='Existing heldout pilot only; conditional document uncertainty, not population certification.'))
    files=list(f for f in P.rglob('*') if f.suffix in ['.py','.md','.json'] and f.name!='manifest.json')
    files+=[ROOT/'causal_direction/data/manifest.json',ROOT/'query_amplitude_ablation/results/benchmark.json']
    for g in GROUPS:
        for seed in SEEDS:
            name=model_name(g,seed)
            base=ROOT/('query_amplitude_ablation' if g=='ad_plain' else 'causal_direction' if g=='favor' else 'hedgehog_matched')
            files += [base/'fits'/f'{name}.pt',base/'fits'/f'{name}.json']
            files += [parent_file(g,seed,split,metric) for split in ['confirm_wiki','confirm_long'] for metric in ['kernel','ppl']]
    save(P/'results/manifest.json',dict(sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(set(files))}))
    print(json.dumps(dict(passed=True,max_metric_errors=errors,toy=toy)),flush=True)

if __name__=='__main__':main()
