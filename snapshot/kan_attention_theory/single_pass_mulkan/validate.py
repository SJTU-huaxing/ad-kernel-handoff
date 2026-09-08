"""Audit data, one-pass metadata, exact parameter budgets, and kernel arithmetic."""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import torch
from models import FeaturePair

P=Path(__file__).resolve().parent


def load_model(path,device='cpu',dtype=torch.float32,indices=None):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    state=saved['state_dict'];meta=saved['metadata']
    if indices is not None:
        state={k:(v if 'knots' in k else v[indices]) for k,v in state.items()}
    norm={k:state[k].to(device=device,dtype=dtype) for k in ['q_mean','q_std','k_mean','k_std']}
    model=FeaturePair(meta['method'],norm,meta['budget_scale']).to(device=device,dtype=dtype)
    model.load_state_dict(state);model.eval()
    return model,meta


def data_audit():
    manifest=json.loads((P/'data/manifest.json').read_text());docs=manifest['documents']
    assert len(set(d['token_sha256'] for d in docs))==len(docs)
    assert len(set(d['text_sha256'] for d in docs))==len(docs)
    assert Counter(d['split'] for d in docs)==Counter(train=4096,validation=128,test=256)
    paired=0
    for shard in manifest['shards']:
        if shard['split']!='train':continue
        batch=torch.load(P/'data'/shard['file'],weights_only=True,map_location='cpu')
        positions=batch['key_positions']
        assert torch.equal(positions.sort(-1).values,torch.arange(512).expand_as(positions))
        records=[d for d in docs if d['shard']==shard['file']]
        for i,d in enumerate(records):
            assert hashlib.sha256(json.dumps(positions[i].tolist()).encode()).hexdigest()==d['pairing_sha256']
        assert batch['q'].shape==batch['k'].shape==(len(records),4,512,128)
        paired+=len(records)*512
    return dict(documents=len(docs),splits=dict(Counter(d['split'] for d in docs)),
                unique_pairs_per_head=paired,all_key_permutations_valid=True,
                all_document_token_and_text_hashes_unique=True,
                manifest_source_note=manifest['source_note'])


@torch.no_grad()
def fit_audit():
    paths=sorted((P/'fits').glob('*.json'));orders=defaultdict(set);checked=[]
    distribution=json.loads((P/'distribution.json').read_text())
    floors=distribution['svd']['documents'];floor_checks=0
    batch=torch.load(P/'data/test_000.pt',weights_only=True,map_location='cpu')
    max_kernel_error=0.;max_attention_error=0.;max_decomposition_error=0.
    for path in paths:
        r=json.loads(path.read_text());n=r['training_documents'];budget=r['budget_scale']
        assert r['epochs']==1 and r['steps']==n and r['unique_pairs_per_head']==n*512
        assert r['loss'] in ['logcosh','poisson']
        assert r['parameters_per_qk_pair']==2*(36864*budget+64)
        for row,floor in zip(r['test_block']['documents'],floors):
            assert row['ordinal']==floor['ordinal']
            for error,energy,bound in zip(row['raw_sse'],row['raw_energy'],floor['residuals']['64']):
                assert error/energy+1e-10>=bound
                floor_checks+=1
        orders[n,r['seed']].add(r['document_order_sha256'])
        model,_=load_model(path.with_suffix('.pt'),dtype=torch.float64)
        assert sum(p.numel() for p in model.parameters())//4==r['parameters_per_qk_pair']
        q=batch['q'][0,:,512:528].double();k=batch['k'][0,:,:24].double()
        lf=model.log_feature(q,'q');lg=model.log_feature(k,'k')
        direct=lf.exp()@lg.exp().transpose(-1,-2)
        ks=lg.amax(1,keepdim=True);qc=lf+ks;qs=qc.amax(-1,keepdim=True)
        scaled=(qc-qs).exp()@(lg-ks).exp().transpose(-1,-2)
        restored=scaled*qs.exp()
        ke=float((restored-direct).norm()/direct.norm());max_kernel_error=max(max_kernel_error,ke)
        a=direct/direct.sum(-1,keepdim=True);b=scaled/scaled.sum(-1,keepdim=True)
        ae=float((a-b).norm()/a.norm());max_attention_error=max(max_attention_error,ae)
        assert ke<1e-12 and ae<1e-12
        log_target=q@k.transpose(-1,-2)/math.sqrt(128)-torch.tensor(r['log_scale'],dtype=torch.float64)[:,None,None]
        truth=log_target.exp();hat=restored
        z=truth.sum(-1);zh=hat.sum(-1);p=truth/z[...,None];ph=hat/zh[...,None]
        lhs=(hat-truth+truth*(log_target-hat.log())).sum(-1)
        rhs=z*(p*(p.log()-ph.log())).sum(-1)+z*(z.log()-zh.log())-z+zh
        de=float((lhs-rhs).norm()/lhs.norm());max_decomposition_error=max(max_decomposition_error,de)
        assert de<1e-10
        checked.append(r['name'])
    assert all(len(v)==1 for v in orders.values())
    assert len(paths)==45
    return dict(runs=len(paths),all_runs_single_pass=True,all_budgets_exact=True,
                orders_identical_across_methods_losses_budgets=True,no_MSE_training=True,
                raw_kernel_restoration_max_relative_l2=max_kernel_error,
                normalized_attention_invariance_max_relative_l2=max_attention_error,
                raw_I_divergence_row_decomposition_max_relative_l2=max_decomposition_error,
                empirical_SVD_lower_bound_checks=floor_checks,empirical_SVD_violations=0,
                checked_runs=checked)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4)
    result={'data':data_audit()}
    if not args.data_only:result['fits']=fit_audit()
    (P/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
