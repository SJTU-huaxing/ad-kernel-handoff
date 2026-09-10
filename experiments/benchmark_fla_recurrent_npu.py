"""Validate/time installed FLA recurrent numerator at current training shapes.

Fixed inputs, no model/optimizer updates. Compare exact normalization outside
FLA, because normalize=True adds epsilon and changes the research kernel.
"""
import hashlib
import json
import math
from pathlib import Path

import torch
import torch_npu
import fla
from fla.ops.linear_attn import fused_recurrent_linear_attn
from ad_kernel.attention import LinearState, _scaled_features, chunk_attention
from benchmark_feature_maps_npu import measure, errors

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/reproduction/feature_speed_npu_v1/recurrent_training_shapes.json'


def recurrent(q,k,v):
    numerator,state=fused_recurrent_linear_attn(q.transpose(1,2).contiguous(),k.transpose(1,2).contiguous(),v.transpose(1,2).contiguous(),
        scale=1.,normalize=False,output_final_state=True)
    den=(q*k.cumsum(2)).sum(-1,keepdim=True)
    return numerator.transpose(1,2)/den,state


def native(logq,logk,v):
    return chunk_attention(logq,logk,v,backend='torch')[0]


def recurrent_with_current_inputs(logq,logk,v):
    # Match the current controller's input, scaling, denominator check and state
    # materialization. This bounded benchmark refuses unsafe denominators; it
    # does not claim to integrate recursive subdivision or initial-state resume.
    q,k,g,_,_=_scaled_features(logq,logk,None,None)
    cumulative=k.cumsum(2)
    den=(q*cumulative).sum(-1,keepdim=True)
    if den.detach().amin().item()<math.sqrt(torch.finfo(q.dtype).tiny):
        raise ValueError('Benchmark input requires numerical subdivision')
    numerator,kv=fused_recurrent_linear_attn(q.transpose(1,2).contiguous(),k.transpose(1,2).contiguous(),v.transpose(1,2).contiguous(),
        scale=1.,normalize=False,output_final_state=True)
    z=cumulative[:,:,-1]
    final_g=torch.where(z>0,g,torch.full_like(g,-torch.inf))
    state=LinearState(kv,z,final_g)
    return numerator.transpose(1,2)/den,state


def main():
    torch.set_num_threads(4)
    torch.npu.set_device('npu:0')
    torch.npu.matmul.allow_hf32=False
    torch.manual_seed(93117)
    report={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Shared FP32 log-feature inputs, dV64; both controllers include the same scaling/denominator guard/final S,z,g materialization; FLA additionally includes BHTM-to-BTHM contiguous conversion. No feature MLP/full model, zero initial state; bounded random inputs do not trigger numerical subdivision.',
        'checks':[],'rows':[],'complete':False}
    def save():OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for m in [64,128,384]:
        # Independent CPU FP64 check includes a chunk boundary and final state.
        xs=[x.to('npu:0').requires_grad_() for x in [torch.randn(1,2,65,m).softmax(-1),torch.randn(1,2,65,m).softmax(-1),torch.randn(1,2,65,64)]]
        refs=[x.detach().cpu().double().requires_grad_() for x in xs]
        qr,kr,vr=refs
        a=(qr@kr.transpose(-1,-2)).tril()
        expected=(a/a.sum(-1,keepdim=True))@vr
        out,st=recurrent(*xs)
        dy=torch.randn_like(out)
        gs=torch.autograd.grad(out,xs,dy)
        gr=torch.autograd.grad(expected,refs,dy.cpu().double())
        check={'m':m,'output':errors(out,expected),'state':errors(st,kr.transpose(-1,-2)@vr),'gradients':[errors(x,y) for x,y in zip(gs,gr)]}
        check['passed']=max(x['relative_rms'] for x in [check['output'],check['state'],*check['gradients']])<1e-4
        report['checks'].append(check);save();print(json.dumps({'event':'CPU_reference_check',**check}),flush=True)
        if not check['passed']:continue
        del xs,refs,out,st,gs,gr,expected
        q,k=[torch.randn(16,12,1024,m).softmax(-1).to('npu:0') for _ in range(2)]
        v=torch.randn(16,12,1024,64).to('npu:0')
        lq,lk=q.log(),k.log()
        q.requires_grad_();k.requires_grad_();v.requires_grad_()
        lq.requires_grad_();lk.requires_grad_()
        dy=torch.randn_like(v)
        a,_=recurrent_with_current_inputs(lq,lk,v);b=native(lq,lk,v)
        ga=torch.autograd.grad(a,(lq,lk,v),dy)
        gb=torch.autograd.grad(b,(lq,lk,v),dy)
        compare={'m':m,'batch':16,'length':1024,'scope':'FLA vs CPU-validated CANN implementation at timing shape',
                 'output':errors(a,b),'gradients':[errors(x,y) for x,y in zip(ga,gb)]}
        compare['passed']=max(x['relative_rms'] for x in [compare['output'],*compare['gradients']])<1e-4
        report['checks'].append(compare);save();print(json.dumps({'event':'full_shape_check',**compare}),flush=True)
        if not compare['passed']:continue
        del a,b,ga,gb
        for ri in range(2):
            for mode in (['fla_recurrent','cann_chunk'] if ri==0 else ['cann_chunk','fla_recurrent']):
                def forward():return recurrent_with_current_inputs(lq,lk,v)[0] if mode=='fla_recurrent' else native(lq,lk,v)
                def backward():
                    o=forward()
                    return torch.autograd.grad(o,(lq,lk,v),dy)
                with torch.no_grad():f=measure(forward,samples=7)
                fb=measure(backward,samples=7)
                row={'m':m,'batch':16,'heads':12,'length':1024,'method':mode,'round':ri,'forward':f,'forward_backward':fb}
                report['rows'].append(row);save()
                print(json.dumps({'event':'timed','m':m,'mode':mode,'round':ri,'forward_us':f['median_us'],'forward_backward_us':fb['median_us']}),flush=True)
        del q,k,v,lq,lk,dy
        torch.npu.empty_cache()
    report['complete']=True;save()


if __name__=='__main__':main()
