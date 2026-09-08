from ablation import *


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    ds=source.data('confirm_long')
    rows=[]
    for regime in ['product_i','causal_i']:
        candidates=[]
        for name in plan()['models']:
            if name.startswith(regime+'_reduced_matched_'):
                result=json.loads((P/'results'/f'kernel_confirm_long_{name}.json').read_text())
                head=max(range(H),key=lambda h:result['summary']['raw_nmse'][h])
                candidates.append((result['summary']['raw_nmse'][head],head,name,result))
        _,head,name,result=max(candidates,key=lambda r:r[0])
        ordinal=max(range(len(result['documents'])),key=lambda i:result['documents'][i]['raw_sse'][head])
        net,meta=load_fit(name,device='cpu')
        for mod in net.modules():
            for key,value in list(mod._parameters.items()):
                if value is not None:mod._parameters[key]=nn.Parameter(value[head:head+1].contiguous(),requires_grad=False)
            for key,value in list(mod._buffers.items()):
                if value is not None and value.ndim and value.shape[0]==H:mod._buffers[key]=value[head:head+1].contiguous()
        net.heads=1
        q=ds['q'][ordinal,head:head+1].double()
        k=ds['k'][ordinal,int(KI[head]):int(KI[head])+1].double()
        positions=ds['query_positions'][ordinal]
        mask=(torch.arange(k.shape[1])[None]<=positions[:,None])[None]
        scale=meta['log_scale'][head]
        lt=q@k.transpose(-1,-2)/math.sqrt(D)-scale
        lp=net.log_matrix(q,k)
        t=lt.masked_fill(~mask,-torch.inf).exp();pred=lp.masked_fill(~mask,-torch.inf).exp()
        errors=(t-pred).square()
        sse=float(errors.sum());saved=result['documents'][ordinal]['raw_sse'][head]
        relative=abs(sse-saved)/max(abs(saved),1e-30)
        assert relative<1e-10,(name,relative)
        flat=int(errors.argmax());qi,ki=divmod(flat,k.shape[1])
        direct=float(torch.logsumexp(net.raw_log_feature(q,'q')[0,qi]+net.raw_log_feature(k,'k')[0,ki],-1))
        discrepancy=abs(direct-float(lp[0,qi,ki]+scale))
        assert discrepancy<1e-10
        pinned=net.raw_log_feature(q,'q').exp()@net.raw_log_feature(net.k_mean[:,None],'k').exp().transpose(-1,-2)
        pin_error=float((pinned-1/64).abs().max())
        assert pin_error<1e-12
        rows.append(dict(regime=regime,name=name,head=HEADS[head],document_ordinal=ordinal,
                         head_raw_nmse=result['summary']['raw_nmse'][head],saved_document_sse=saved,
                         cpu_fp64_document_sse=sse,relative_recomputation_error=relative,
                         worst_query_position=int(positions[qi]),worst_key_position=ki,
                         worst_target_raw_log_kernel=float(lt[0,qi,ki]+scale),worst_prediction_raw_log_kernel=direct,
                         worst_pair_share_of_document_sse=float(errors[0,qi,ki]/errors.sum()),
                         direct_logsumexp_discrepancy=discrepancy,at_key_mean_kernel_one_over_m_error=pin_error))
    save(P/'checks'/'raw_cases.json',dict(passed=True,rows=rows,
         scope='Post-hoc independent CPU FP64 audit of worst long-document errors, no training or model selection. The key-mean identity is algebraic; mean key need not be on real support.'))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
