"""Variable-rank local replacement with one actual state per head (no padding)."""
from core import *
sys.path.insert(0,str(ROOT/'deployment_validation'))
from runtime import causal_prefill,recurrent_step

class HeadMap:
    def __init__(self,name,head,dtype=torch.float32):
        net,meta=load_fit(name,dtype)
        self.raw_kernel_name=name;self.equivalent_parent_name=None
        if meta['kind']=='gauge_calibrated':
            # The calibrated raw kernel is a(q) times this parent kernel.
            # Compile the exactly cancelled row scalar away by executing the
            # parent's original factors. This also avoids FP32 changes from
            # folding feature-wise mu rescalings into both factors. Only one
            # parameter set is retained in memory, with the same budget.
            self.equivalent_parent_name=f"exp_control_m64_s{meta['seed']}_n4096"
            net,_=load_fit(self.equivalent_parent_name,dtype)
        for mod in net.modules():
            for k,v in list(mod._parameters.items()):
                if v is not None:mod._parameters[k]=nn.Parameter(v[head:head+1].contiguous(),requires_grad=False)
            for k,v in list(mod._buffers.items()):
                if v is not None and v.ndim and v.shape[0]==4:mod._buffers[k]=v[head:head+1].contiguous()
        self.net=net;self.m=meta['m'];self.dtype=dtype
    def features(self,x,side):
        if side=='q' and self.net.variant in ['factorized','factorized_both','exp','gauge_calibrated']:
            z=self.net.qnet((x.to(self.dtype)-self.net.q_mean[:,None])/self.net.q_std[:,None])
            if self.net.variant!='exp':z=torch.cat([z[...,:-1],torch.zeros_like(z[...,-1:])],-1)
            # Directly omit the exactly cancelled Q amplitude during attention.
            return z.softmax(-1)
        logf=self.net.log_feature(x.to(self.dtype),side)
        # Query-only amplitude cancels in numerator/denominator; raw evaluation
        # restores it. No key/time-dependent rescaling is dropped.
        if self.net.variant in ['factorized','factorized_both','exp'] and side=='q':logf=logf-logf.logsumexp(-1,keepdim=True)
        return logf.exp()

class Replacement:
    def __init__(self,original,names,dtype=torch.float32):
        self.original=original;self.maps={};self.states={};self.diagnostics=[];self.collect_diagnostics=True
        for h,name in enumerate(names):self.maps[tuple(HEADS[h])]=HeadMap(name,h,dtype)
    def reset(self):self.states={};self.diagnostics=[]
    def __call__(self,module,query,key,value,attention_mask,**kwargs):
        layer=module.layer_idx
        if layer not in [14,27]:return self.original(module,query,key,value,attention_mask,**kwargs)
        assert query.shape[0]==1
        n=query.shape[2];total=key.shape[2];remaining=[1,2,3,4,5,7,8,9,10,11]
        exact=F.scaled_dot_product_attention(query[:,remaining],key[:,[h//6 for h in remaining]],value[:,[h//6 for h in remaining]],attn_mask=attention_mask,
            dropout_p=0.,is_causal=n>1 and attention_mask is None,scale=kwargs.get('scaling'))
        out=torch.empty_like(query);out[:,remaining]=exact
        for qhead,kvhead in [(0,0),(6,1)]:
            tag=(layer,qhead);f=self.maps[tag]
            q=query[0,qhead:qhead+1].to(f.dtype)
            k=key[0,kvhead:kvhead+1,-n:].to(f.dtype);v=value[0,kvhead:kvhead+1,-n:].to(f.dtype)
            fq,fk=f.features(q,'q'),f.features(k,'k')
            if n==total:y,den,state=causal_prefill(fq,fk,v);self.states[tag]=state
            else:
                assert n==1 and tag in self.states
                y,den=recurrent_step(fq,fk,v,self.states[tag])
            if self.collect_diagnostics:self.diagnostics.append(torch.stack([(den<=0).sum(),(~torch.isfinite(y).all(-1)).sum(),torch.tensor(den.numel(),device=den.device)]))
            out[0,qhead:qhead+1]=y.to(query.dtype)
        return out.transpose(1,2).contiguous(),None
