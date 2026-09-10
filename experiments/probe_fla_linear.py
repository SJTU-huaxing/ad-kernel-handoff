"""Determine whether the installed FLA numerator is usable for pure linear attention."""
import argparse
import json
import time
import torch
import torch_npu
import fla
from fla.ops.linear_attn import chunk_linear_attn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--length", type=int, default=65)
    parser.add_argument("--backend", choices=["fla", "ascend-hybrid"], default="fla")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(921)
    dtype = getattr(torch, args.dtype)
    t = args.length
    values = [torch.randn(1, t, 2, 64).softmax(-1),
              torch.randn(1, t, 2, 64).softmax(-1), torch.randn(1, t, 2, 64)]
    inputs = [x.to(dtype).to(args.device).requires_grad_() for x in values]
    refs = [x.detach().cpu().double().requires_grad_() for x in inputs]
    q, k, v = inputs
    qr, kr, vr = [x.transpose(1, 2) for x in refs]
    attention = (qr @ kr.transpose(-1, -2)).tril()
    ref = ((attention / attention.sum(-1, keepdim=True)) @ vr).transpose(1, 2)
    print(json.dumps({"event": "start", "dtype": args.dtype, "length": t}), flush=True)
    start = time.perf_counter()
    if args.backend == "fla":
        numerator, state = chunk_linear_attn(q, k, v, scale=1.0, normalize=False, output_final_state=True)
    else:
        from ad_kernel.ascend_linear import ascend_linear_numerator
        numerator, state = ascend_linear_numerator(q, k, v)
    # FLA's normalize=True adds epsilon; compute the exact positive denominator
    # in FP32 here to preserve the project's mathematical normalization.
    denom = (q.float() * k.float().cumsum(1)).sum(-1, keepdim=True)
    output = numerator.float() / denom
    output_before = output.detach().cpu().double().clone()
    numerator_before = numerator.detach().cpu().double().clone()
    input_before = [x.detach().cpu().clone() for x in inputs]
    print(json.dumps({"event":"forward","seconds":time.perf_counter()-start,
                      "output_max_abs":(output_before-ref.detach()).abs().max().item(),
                      "per_token_max_abs":(output_before-ref.detach()).abs().amax((0,2,3)).tolist()}),flush=True)
    dy = torch.randn_like(output).to(dtype)
    output.backward(dy)
    ref.backward(dy.cpu().double())
    torch.npu.synchronize()
    result = {"event": "complete", "dtype": args.dtype, "length": t, "backend": args.backend,
              "seconds": time.perf_counter()-start,
              "output_max_abs": (output.cpu().double()-ref).abs().max().item(),
              "output_modified_by_backward":(output.detach().cpu().double()-output_before).abs().max().item(),
              "input_modified_by_forward_backward":[(a.detach().cpu()-b).abs().max().item() for a,b in zip(inputs,input_before)],
              "gradient_relative_rms": {}}
    for name, actual, expected in zip(["q", "k", "v"], inputs, refs):
        error = (actual.grad.cpu().double()-expected.grad).square().mean().sqrt()
        norm = expected.grad.square().mean().sqrt()
        result["gradient_relative_rms"][name] = (error/norm).item()
    print(json.dumps(result), flush=True)
    assert result["output_max_abs"] < (1e-4 if dtype == torch.float32 else 0.03), result
    assert max(result["gradient_relative_rms"].values()) < (1e-4 if dtype == torch.float32 else 0.03), result


if __name__ == "__main__":
    main()
