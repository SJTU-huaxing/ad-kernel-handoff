import hashlib
from core_attribution import *

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    torch.set_num_threads(4)
    models=plan()['models'];assert len(models)==9 and len(list((P/'fits').glob('*.json')))==12
    checks={x:read(P/'checks'/f'{x}.json') for x in ['structure','mechanism','operators','long_precision']}
    assert all(x['passed'] for x in checks.values())
    assert len(checks['operators']['rows'])==9 and len(checks['long_precision']['rows'])==3
    protocol=sha(P/'PROTOCOL.zh.md');groups=all_groups();rows=[]
    oldnorm=None
    for name in models:
        fit=read(file_for(name,'fit'));meta=fit['metadata'];seed=meta['seed']
        old=read(file_for(old_name(seed),'fit'))['metadata']
        for field in ['heads','m','parameters_per_head','train_documents','training_queries_per_head','training_pairs_per_head','epochs','order_sha256','query_sha256','log_scale']:
            assert meta[field]==old[field],(name,field)
        assert meta['protocol_sha256']==protocol
        net,_=load_fit(name,torch.float32,'cpu')
        ref,_=load_fit(old_name(seed),torch.float32,'cpu')
        for key in ['q_mean','q_std','k_mean','k_std','numerical_scale']:
            assert torch.equal(getattr(net,key),getattr(ref,key)),(name,key)
        norm={key:getattr(net,key) for key in ['q_mean','q_std','k_mean','k_std']}
        torch.manual_seed(seed);initial=Features(norm,meta['map_kind'],meta['initialization'],net.numerical_scale)
        assert fingerprint(initial.qnet.state_dict())==meta['q_initial_sha256']
        assert fingerprint(initial.knet.state_dict())==meta['k_initial_sha256']
        assert sum(p.numel() for p in net.parameters())==24*73536
        assert net.qnet.w2.shape==(24,63,192) and net.knet.w2.shape==(24,64,192)
        rows.append(dict(name=name,checkpoint_sha256=sha(file_for(name,'fit','pt')),
            order_sha256=meta['order_sha256'],query_sha256=meta['query_sha256'],
            q_initial_sha256=meta['q_initial_sha256'],k_initial_sha256=meta['k_initial_sha256']))
    for kind in ['ad','exp']:
        for seed in SEEDS:
            names=[n for n in models if f'matched_zero_{kind}_s{seed}_' in n]
            assert len(names)==1
    for seed in SEEDS:
        names=[n for n in models if f'matched_zero_' in n and f'_s{seed}_' in n]
        mm=[read(file_for(n,'fit'))['metadata'] for n in names]
        for key in ['q_initial_sha256','k_initial_sha256']:assert mm[0][key]==mm[1][key]
    for split,count,n in [('confirm_wiki',128,1024),('confirm_long',24,8192)]:
        diag=read(P/'results'/f'diagnostics_{split}.json');assert len(diag['models'])==12
        assert diag['max_discrepancy_with_parent_metrics']<1e-9
        for name in [n for names in groups.values() for n in names]:
            k=read(file_for(name,f'kernel_{split}'));p=read(file_for(name,f'ppl_{split}'))
            assert len(k['documents'])==count and len(p['documents'])==count
            assert sum(r['tokens'] for r in p['documents'])==count*(n-1)
            assert all(math.isfinite(v) for v in k['summary']['kl'])
            assert abs(math.exp(sum(r['nll'] for r in p['documents'])/(count*(n-1)))-p['ppl'])<1e-10
    summary=read(P/'results/summary.json')
    assert all(abs(r['difference'])<1e-9 for r in summary['teacher_check'])
    for g,names in groups.items():
        trials=[r for r in summary['selection'] if r['group']==g];assert len(trials)==2
        chosen=min(trials,key=lambda x:x['validation_kl']);assert chosen['selected']
        assert read(file_for(names[0],'fit'))['metadata']['lr']==chosen['lr']
    assert all(r['state_bytes_per_layer']==399360 for r in summary['benchmark'])
    assert (P/'REPORT.zh.md').exists()
    source_files=list(P.glob('*.py'))+[P/'PROTOCOL.zh.md',P/'REPORT.zh.md',
        ROOT/'query_amplitude_ablation/ablation.py',ROOT/'query_amplitude_ablation/train_run.py',
        ROOT/'hedgehog_matched/models.py',ROOT/'hedgehog_matched/assess.py',
        ROOT/'causal_direction/common.py',ROOT/'causal_direction/data/manifest.json']
    # Resolve the exact state implementation used by the parent runtime.
    import operators
    source_files.append(__import__('pathlib').Path(operators.__file__))
    save(P/'checks/audit.json',dict(passed=True,new_fit_count=12,new_selected_count=9,reused_selected_count=3,
        checks=list(checks),metadata_and_parameter_checks=rows,
        source_sha256={str(p):sha(p) for p in source_files},
        result_sha256={str(p):sha(p) for p in (P/'results').glob('*.json')},
        scope='Same cached data IDs/order/positions/normalization, parameter count and initial weights, validation-only LR selection, operator precision and PPL arithmetic checked. No population generalization certification.'))
    print(json.dumps(dict(passed=True,new_fits=12,compared_models=12)),flush=True)

if __name__=='__main__':main()
