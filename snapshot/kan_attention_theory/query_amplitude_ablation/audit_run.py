import hashlib
from ablation import *


def main():
    selected = plan()['models']
    fits = list((P/'fits').glob('*.json'))
    assert len(fits)==24 and len(selected)==18
    rows=[]
    for name in selected:
        meta=json.loads((P/'fits'/f'{name}.json').read_text())['metadata']
        ref=old_name(meta['regime'],'ad',meta['seed'])
        original=json.loads((OLD/'fits'/f'{ref}.json').read_text())['metadata']
        for field in ['order_sha256','query_sha256','training_pairs_per_head','training_queries_per_head','train_documents','epochs']:
            assert meta[field]==original[field],(name,field)
        assert meta['log_scale']==original['log_scale'],name
        net,_=load_fit(name,device='cpu')
        assert net.qnet.w2.shape[1]==63
        assert not hasattr(net,'log_scale')
        assert sum(p.numel() for p in net.parameters())//H==meta['parameters_per_head']
        for split in ['confirm_wiki','confirm_long']:
            for prefix in ['kernel','ppl']:
                assert (P/'results'/f'{prefix}_{split}_{name}.json').exists()
        for direction in ['ab','ba']:
            assert (P/'results'/f'product_{direction}_{name}.json').exists()
        rows.append(dict(name=name,params=meta['parameters_per_head'],matched_parent_protocol=ref))
    for stem in ['structure','operators','long_precision','raw_cases']:
        assert json.loads((P/'checks'/f'{stem}.json').read_text())['passed']
    assert len(json.loads((P/'checks'/'operators.json').read_text())['rows'])==18
    assert len(json.loads((P/'checks'/'long_precision.json').read_text())['rows'])==6
    assert len(json.loads((P/'results'/'benchmark.json').read_text())['aggregates'])==12
    teacher=[]
    for split in ['confirm_wiki','confirm_long']:
        new=json.loads((P/'results'/f'ppl_{split}_teacher.json').read_text())
        old=json.loads((OLD/'results'/f'ppl_{split}_teacher.json').read_text())
        error=abs(new['ppl']-old['ppl'])
        assert error<1e-5,(split,error)
        teacher.append(dict(split=split,ppl=new['ppl'],absolute_difference=error))
    save(P/'checks'/'final_audit.json',dict(passed=True,rows=rows,teacher=teacher,
        budgets='Matched rawI73729 and KL73728; pure deletion73536. Original KL AD contains192 query-amplitude weights invisible to KL.',
        scope='No certificate for population optimality or new blind testing.'))
    sources=[ROOT/'causal_direction/data/manifest.json']
    for regime in REGIMES:
        for kind in ['ad','hh']+(['hh_softmax'] if regime=='causal_kl' else []):
            for seed in SEEDS:
                name=old_name(regime,kind,seed)
                sources.extend([OLD/'fits'/f'{name}.json',OLD/'fits'/f'{name}.pt'])
                sources.extend(OLD/'results'/f'{prefix}_{split}_{name}.json' for prefix in ['kernel','ppl'] for split in ['confirm_wiki','confirm_long'])
                sources.extend(OLD/'results'/f'product_{direction}_{name}.json' for direction in ['ab','ba'])
    local=[f for f in P.rglob('*') if f.suffix in ['.py','.md','.json','.pt','.png','.pdf'] and f.name!='manifest.json']
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(set(sources+local))}
    save(P/'results'/'manifest.json',dict(sha256=hashes))
    print(json.dumps(dict(passed=True,fits=len(fits),selected=len(selected),teacher=teacher)),flush=True)


if __name__=='__main__':main()
