"""Exhaustive raw-kernel risk over entire new empirical marginal products.

Three base kernels, with gauge-derived variants handled analytically to avoid
repeating dense multiplications. All computations FP64; seed11 fixed in advance.
"""
from core import *

@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    groups=['raw','factorized_both','exp_control','favor_plus']
    for ds in ['fresh_wiki3','swde3']:
        path=P/'results'/f'product_{ds}.json'
        if path.exists():continue
        data=torch.load(P/'results'/f'data_{ds}.pt',weights_only=True);mid=data['q'].shape[2]//2
        q=data['q'][:,:,mid:].permute(1,0,2,3).flatten(1,2).cuda().double()
        k=data['k'][:,:,:mid].permute(1,0,2,3).flatten(1,2).cuda().double();n=q.shape[1]
        qs={};ks={};logs={};metas={}
        for kind in groups+['gauge_calibrated','global_bal','global_raw']:
            if kind=='favor_plus':
                from baselines import RF
                meta=dict(metas['raw'],rf_seed=1009);net=RF('favor_plus',1009,meta['log_scale'])
            else:net,meta=load_fit(f'{kind}_m64_s11_n4096')
            metas[kind]=meta
            lq=torch.cat([net.log_feature(q[:,i:i+1024],'q') for i in range(0,n,1024)],1)
            if kind in groups:
                qs[kind]=lq.exp();ks[kind]=torch.cat([net.log_feature(k[:,i:i+1024],'k').exp() for i in range(0,n,1024)],1)
                logs[kind]=torch.zeros(4,n,device='cuda',dtype=torch.float64)
                if kind=='exp_control':base_lq=lq
            elif kind=='gauge_calibrated':
                mu=torch.tensor(meta['log_key_feature_means'],device='cuda',dtype=torch.float64)
                differences=lq-base_lq-mu[:,None]
                assert (differences-differences.mean(-1,keepdim=True)).abs().max()<1e-9
                logs[kind]=differences.mean(-1)
            else:logs[kind]=(lq-base_lq).mean(-1)
        scale=torch.tensor(metas['raw']['log_scale'],device='cuda',dtype=torch.float64)
        cases=list(logs);group={c:c if c in groups else 'exp_control' for c in cases}
        predmass={};predenergy={}
        for c in cases:
            fq=qs[group[c]]*logs[c].exp()[:,:,None];fk=ks[group[c]]
            predmass[c]=(fq.sum(1)*fk.sum(1)).sum(-1)
            predenergy[c]=((fq.transpose(-1,-2)@fq)*(fk.transpose(-1,-2)@fk)).sum((-1,-2))
        energy=torch.zeros(4,device='cuda',dtype=torch.float64);mass=energy.clone();entropy=energy.clone()
        cross={c:energy.clone() for c in cases};crosslog={g:energy.clone() for g in groups}
        rowmass=torch.zeros(4,n,device='cuda',dtype=torch.float64);rowentropy=rowmass.clone()
        rowcross={g:rowmass.clone() for g in groups};start=time.perf_counter()
        for qi in range(0,n,1024):
            qq=q[:,qi:qi+1024];end=qi+qq.shape[1]
            for ki in range(0,n,4096):
                kk=k[:,ki:ki+4096];kend=ki+kk.shape[1]
                logt=qq@kk.transpose(-1,-2)/math.sqrt(128)-scale[:,None,None];t=logt.exp()
                energy+=t.square().sum((-1,-2));mass+=t.sum((-1,-2));entropy+=(t*logt).sum((-1,-2))
                rowmass[:,qi:end]+=t.sum(-1);rowentropy[:,qi:end]+=(t*logt).sum(-1)
                for g in groups:
                    pred=qs[g][:,qi:end]@ks[g][:,ki:kend].transpose(-1,-2)
                    weighted_log=t*pred.clamp_min(1e-300).log()
                    crosslog[g]+=weighted_log.sum((-1,-2));rowcross[g][:,qi:end]+=weighted_log.sum(-1)
                    products=(t*pred).sum(-1)
                    for c in cases:
                        if group[c]==g:cross[c]+=(products*logs[c][:,qi:end].exp()).sum(-1)
            if qi%8192==0:print(json.dumps(dict(event='product',dataset=ds,queries=end,total=n,seconds=time.perf_counter()-start)),flush=True)
        rows=[]
        for c in cases:
            g=group[c];sse=energy-2*cross[c]+predenergy[c]
            div=entropy-crosslog[g]-(rowmass*logs[c]).sum(-1)-mass+predmass[c]
            base_row=(qs[g]*ks[g].sum(1)[:,None]).sum(-1)
            kl=(rowentropy-rowcross[g])/rowmass-rowmass.log()+base_row.log()
            ratio_log=base_row.log()+logs[c]-rowmass.log()
            row_div=kl+torch.expm1(ratio_log)-ratio_log
            rows.append(dict(method=c,seed=1009 if c=='favor_plus' else 11,relative_squared_risk=(sse/energy).tolist(),relative_I_divergence=(div/mass).tolist(),
                query_balanced_divergence=row_div.mean(-1).tolist(),directional_kl=kl.mean(-1).tolist(),
                mass_divergence=(row_div-kl).mean(-1).tolist()))
        save(path,dict(dataset=ds,nq=n,nk=n,pairs_per_head=n*n,results=rows,seconds=time.perf_counter()-start,
            log_truth_energy=(energy.log()-2*math.log(n)+2*scale).tolist(),
            scope='Exact full empirical marginal product, all new confirmation vectors; no matrix fitting; FP64. Does not constitute an unknown-population confidence bound. Fixed seed11, not selected by test.'))
        print(json.dumps(dict(event='product_complete',dataset=ds,results=rows)),flush=True)

if __name__=='__main__':main()
