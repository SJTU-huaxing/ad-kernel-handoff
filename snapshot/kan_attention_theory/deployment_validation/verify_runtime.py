"""Independent dense-causal and recurrent-prefix verification on real Q/K/V."""
import json
import torch
from runtime import P,FeatureMap,causal_prefill,recurrent_step

@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    raw=torch.load(P.parent/'single_pass_mulkan/data/test_000.pt',weights_only=True)
    q,k,v=[raw[s][0].cuda().float() for s in ['q','k','v']]
    rows=[]
    for method in ['partition','galerkin','favor_plus','favor_plus_640']:
        f64=FeatureMap(method,dtype=torch.float64,optimized=False)
        f32=FeatureMap(method,dtype=torch.float32,optimized=True)
        if method=='partition':
            for side,x in [('q',q),('k',k)]:
                a=f64.cells(x.double(),side);b=f32.cells(x,side)
                assert torch.equal(a,b)
        a,b,gauge=f64.prefill(q.double(),k.double())
        dense=(a@b.transpose(-1,-2)).tril();reference=(dense/dense.sum(-1,keepdim=True))@v.double()
        y,den,state=causal_prefill(a,b,v.double())
        err=float((y-reference).norm()/reference.norm());assert err<1e-8,err
        a32,b32,_=f32.prefill(q,k);y32,_,_=causal_prefill(a32,b32,v)
        precision=float((y32.double()-y).norm()/y.norm())
        aa,bb,gg=f64.prefill(q[:,:1000].double(),k[:,:1000].double())
        _,_,state=causal_prefill(aa,bb,v[:,:1000].double());state['gauge']=gg
        ys=[]
        for t in range(1000,1024):
            aq,bk=f64.decode(q[:,t:t+1].double(),k[:,t:t+1].double(),state)
            yy,_=recurrent_step(aq,bk,v[:,t:t+1].double(),state,method=='partition');ys.append(yy)
        ys=torch.cat(ys,1);err_decode=float((ys-y[:,1000:]).norm()/y[:,1000:].norm())
        assert err_decode<1e-8,err_decode
        # Test query suffix cannot depend on future keys by prefix comparison.
        ap,bp,_=f64.prefill(q[:,:512].double(),k[:,:512].double())
        yp,_,_=causal_prefill(ap,bp,v[:,:512].double())
        err_causal=float((yp-y[:,:512]).norm()/y[:,:512].norm());assert err_causal<1e-8
        aa,bb,gg=f32.prefill(q[:,:1000],k[:,:1000]);_,_,state32=causal_prefill(aa,bb,v[:,:1000]);state32['gauge']=gg
        ys32=[]
        for t in range(1000,1024):
            aq,bk=f32.decode(q[:,t:t+1],k[:,t:t+1],state32)
            yy,_=recurrent_step(aq,bk,v[:,t:t+1],state32,method=='partition');ys32.append(yy)
        fp32_decode=float((torch.cat(ys32,1).double()-y[:,1000:]).norm()/y[:,1000:].norm())
        assert fp32_decode<1e-3,fp32_decode
        row=dict(method=method,dense_relative_l2=err,recurrent_relative_l2=err_decode,
                 prefix_relative_l2=err_causal,fp32_relative_l2=precision,fp32_recurrent_relative_l2=fp32_decode,
                 nonfinite_fp32=int((~torch.isfinite(y32)).sum()),nonpositive_denominators=int((den<=0).sum()))
        rows.append(row);print(json.dumps(row),flush=True)
    (P/'results').mkdir(exist_ok=True)
    (P/'results/runtime_checks.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
