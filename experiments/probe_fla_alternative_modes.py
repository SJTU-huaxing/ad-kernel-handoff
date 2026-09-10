"""Check unmodified installed FLA alternatives against CPU FP64; no training."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

import torch
import torch_npu
import fla
from fla.ops.linear_attn import fused_chunk_linear_attn, fused_recurrent_linear_attn

ROOT=Path(__file__).resolve().parents[1]


def error(a,b):
    a,b=a.detach().cpu().double(),b.detach().cpu().double()
    return {'max_abs':(a-b).abs().max().item(),'relative_rms':((a-b).square().mean()/b.square().mean().clamp_min(1e-50)).sqrt().item()}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['fused_chunk','fused_recurrent','model','model_recurrent'],required=True)
    ap.add_argument('--dtype',choices=['float32','bfloat16'],default='float32')
    args=ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(921)
    torch.npu.set_device('npu:0')
    dtype=getattr(torch,args.dtype)
    row={'mode':args.mode,'dtype':args.dtype,'length':65,'torch':str(torch.__version__),'torch_npu':str(torch_npu.__version__),
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'passed':False}
    target=ROOT/'work/reproduction/feature_speed_npu_v1'/f'fla_{args.mode}_{args.dtype}.json'
    try:
        if args.mode.startswith('model'):
            from fla.models.linear_attn import LinearAttentionConfig,LinearAttentionForCausalLM
            config=LinearAttentionConfig(hidden_size=128,num_heads=2,num_hidden_layers=1,intermediate_size=256,
                vocab_size=512,feature_map='hedgehog',attn_mode='fused_recurrent' if args.mode=='model_recurrent' else 'fused_chunk',fuse_cross_entropy=False,
                fuse_linear_cross_entropy=False,use_cache=False)
            model=LinearAttentionForCausalLM(config).to('npu:0')
            ids=torch.randint(512,(1,65),device='npu:0')
            with torch.autocast('npu',dtype=torch.bfloat16,enabled=dtype==torch.bfloat16):
                result=model(ids,labels=ids,use_cache=False)
            result.loss.backward()
            torch.npu.synchronize()
            row.update(loss=result.loss.item(),finite_gradients=all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None),
                       scope='One-layer random-weight official FLA LinearAttentionForCausalLM smoke; finite loss/gradients only, not a correctness proof or pretraining run')
            row['passed']=bool(torch.isfinite(result.loss).item() and row['finite_gradients'])
        else:
            inputs=[x.to(dtype).to('npu:0').requires_grad_() for x in [torch.randn(1,65,2,64).softmax(-1),torch.randn(1,65,2,64).softmax(-1),torch.randn(1,65,2,64)]]
            refs=[x.detach().cpu().double().requires_grad_() for x in inputs]
            q,k,v=inputs
            qr,kr,vr=[x.transpose(1,2) for x in refs]
            attn=(qr@kr.transpose(-1,-2)).tril()
            expected=((attn/attn.sum(-1,keepdim=True))@vr).transpose(1,2)
            fn=fused_chunk_linear_attn if args.mode=='fused_chunk' else fused_recurrent_linear_attn
            numerator,state=fn(q,k,v,scale=1.,normalize=False,output_final_state=True)
            out=numerator.float()/(q.float()*k.float().cumsum(1)).sum(-1,keepdim=True)
            dy=torch.randn_like(out)
            out.backward(dy);expected.backward(dy.cpu().double())
            torch.npu.synchronize()
            row.update(output=error(out,expected),gradients={n:error(a.grad,b.grad) for n,a,b in zip(['q','k','v'],inputs,refs)},
                       state=error(state,kr.transpose(-1,-2)@vr),scope='Unmodified FLA numerator and backward; exact positive denominator outside kernel, CPU FP64 reference')
            tolerance=1e-4 if dtype==torch.float32 else .03
            row['passed']=row['output']['max_abs']<tolerance and all(v['relative_rms']<tolerance for v in row['gradients'].values()) and row['state']['relative_rms']<tolerance
    except Exception as e:
        row.update(exception_type=type(e).__name__,exception=str(e)[-10000:])
        traceback.print_exc()
    target.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
    print(json.dumps(row),flush=True)


if __name__=='__main__':main()
