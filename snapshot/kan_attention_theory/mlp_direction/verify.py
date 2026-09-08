from core import *
from runtime_new import HeadMap,causal_prefill,recurrent_step

@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    data=torch.load(P/'results/data_fresh_wiki2.pt',weights_only=True);checks=[]
    cases=[f'half_m{m}_s11_n4096' for m in [16,32,64,96,128]]+[f'{kind}_m64_s{s}_n4096' for kind in ['factorized','factorized_both','exp_control','gauge_calibrated'] for s in [11,29,47]]
    for name in cases:
        net,meta=load_fit(name);q=data['q'][0,:,:37].cuda().double();k=data['k'][0,:,:37].cuda().double();v=data['v'][0,:,:37].cuda().double()
        fq=net.log_feature(q,'q').exp();fk=net.log_feature(k,'k').exp()
        mat=(fq@fk.transpose(-1,-2)).tril();explicit=mat@v/mat.sum(-1,keepdim=True)
        scan,den,state=causal_prefill(fq,fk,v)
        prefix,_,state=causal_prefill(fq[:,:21],fk[:,:21],v[:,:21]);parts=[prefix]
        for t in range(21,37):
            y,_=recurrent_step(fq[:,t:t+1],fk[:,t:t+1],v[:,t:t+1],state,optimized=False);parts.append(y)
        recurrent=torch.cat(parts,1)
        err=float((scan-explicit).norm()/explicit.norm());re=float((recurrent-explicit).norm()/explicit.norm())
        assert err<1e-11 and re<1e-11
        gauges=[]
        for h in range(4):
            hm=HeadMap(name,h,torch.float64);lq=hm.features(q[h:h+1],'q');lk=hm.features(k[h:h+1],'k')
            yy,_,_=causal_prefill(lq,lk,v[h:h+1]);ge=float((yy-explicit[h:h+1]).norm()/explicit[h:h+1].norm())
            assert ge<1e-11;gauges.append(ge)
        checks.append(dict(name=name,scan_relative_l2=err,recurrent_relative_l2=re,head_runtime_relative_l2=gauges))
    m=json.loads((DATA/'manifest.json').read_text());splitsets={s:{r['token_sha256'] for r in m['documents'] if r['split']==s} for s in ['train','validation','test']}
    old=json.loads((ROOT/'real_llm_pilot/data_qwen25_1p5b/manifest.json').read_text())
    splitsets['official']={r['token_sha256'] for r in old['documents'] if r['split']=='test'}
    for f in ['fresh_manifest.json','fresh_manifest2.json','fresh_manifest3.json']:
        records=json.loads((P/'results'/f).read_text())['records']
        for ds in {r['dataset'] for r in records}:splitsets[ds]={r['token_sha256'] for r in records if r['dataset']==ds}
    names=list(splitsets)
    for i,a in enumerate(names):
        for b in names[i+1:]:assert not splitsets[a]&splitsets[b],(a,b)
    fits=[json.loads(p.read_text())['metadata'] for p in (P/'fits').glob('*.json')]
    assert all(r['parameters_per_head']==49152+386*r['m'] and r['epochs']==1 for r in fits)
    for seed in [11,29,47]:
        rows=[r for r in fits if r['seed']==seed]
        assert len({r['document_order_sha256'] for r in rows})==1
        assert len({r['query_indices_sha256'] for r in rows})==1
    save(P/'checks/implementation.json',dict(checks=checks,disjoint_token_hashes=True,split_counts={k:len(v) for k,v in splitsets.items()},
        all_parameter_budgets_correct=True,matched_single_pass_document_and_query_orders=True))
    print(json.dumps(dict(event='verified',cases=len(checks),max_error=max(r['recurrent_relative_l2'] for r in checks),fits=len(fits))),flush=True)

if __name__=='__main__':main()
