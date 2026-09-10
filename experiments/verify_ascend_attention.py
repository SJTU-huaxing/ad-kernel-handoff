"""Check the actual normalized NPU autograd path, tails and state gradients."""
import argparse
import json
from pathlib import Path
import time

import torch
import torch_npu
import fla
from ad_kernel.attention import chunk_attention, dense_attention
from ad_kernel.ascend_linear import ascend_linear_numerator

ROOT = Path(__file__).resolve().parents[1]


def relative_rms(actual, expected):
    return ((actual - expected).square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-7)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], required=True)
    ap.add_argument("--device", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    dtype = getattr(torch, args.dtype)
    tolerance = 5e-4 if dtype == torch.float32 else .05
    results, start = [], time.perf_counter()
    for t, m in [(63, 64), (64, 64), (65, 64), (129, 64), (1024, 64), (65, 128), (65, 256), (65, 384)]:
        torch.manual_seed(9030 + t + m)
        # FP32 log features retain exp/softmax accuracy; numerator uses compute dtype.
        values = [torch.randn(2, 2, t, width) for width in [m, m, 64]]
        inputs = [x.to(args.device).requires_grad_() for x in values]
        refs = [x.double().requires_grad_() for x in values]
        valid = torch.ones(2, t, dtype=torch.bool)
        valid[0, -3:] = False
        actual, state = chunk_attention(*inputs, valid=valid.to(args.device), backend="ascend-hybrid", compute_dtype=dtype)
        expected = dense_attention(*refs, valid=valid)
        dy = torch.randn_like(values[2])
        actual.backward(dy.to(args.device))
        expected.backward(dy.double())
        grads = [relative_rms(x.grad.cpu().double(), r.grad) for x, r in zip(inputs, refs)]
        error = relative_rms(actual.detach().cpu().double(), expected.detach())
        result = {"event": "normalized", "length": t, "m": m, "output_relative_rms": error,
                  "gradient_relative_rms": grads, "seconds": time.perf_counter() - start}
        print(json.dumps(result), flush=True)
        assert error < tolerance and max(grads) < tolerance, result
        results.append(result)
    # Test gradients of a nonzero initial state and a loss on the final state.
    torch.manual_seed(194)
    values = [torch.randn(2, 65, 2, 64).softmax(-1) for _ in range(2)]
    values += [torch.randn(2, 65, 2, 64), torch.randn(2, 2, 64, 64) * .01]
    inputs = [x.to(dtype if i < 3 else torch.float32).to(args.device).requires_grad_() for i, x in enumerate(values)]
    refs = [x.detach().cpu().double().requires_grad_() for x in inputs]
    out, final = ascend_linear_numerator(*inputs)
    q, k, v = [x.transpose(1, 2) for x in refs[:3]]
    h0 = refs[3]
    expected = (q @ h0 + (q @ k.transpose(-1, -2)).tril() @ v).transpose(1, 2)
    final_expected = h0 + k.transpose(-1, -2) @ v
    dy, df = torch.randn_like(out), torch.randn_like(final)
    torch.autograd.backward([out, final], [dy, df])
    torch.autograd.backward([expected, final_expected], [dy.cpu().double(), df.cpu().double()])
    grads = [relative_rms(x.grad.cpu().double(), r.grad) for x, r in zip(inputs, refs)]
    result = {"event": "initial_and_final_state", "gradient_relative_rms": grads,
              "output_relative_rms": relative_rms(out.detach().cpu().double(), expected.detach()),
              "final_state_relative_rms": relative_rms(final.detach().cpu().double(), final_expected.detach())}
    print(json.dumps(result), flush=True)
    assert max(grads + [result["output_relative_rms"], result["final_state_relative_rms"]]) < tolerance
    results.append(result)
    path = ROOT / f"work/reproduction/ascend_attention_{args.dtype}.json"
    path.write_text(json.dumps({"passed": True, "backend": "ascend-hybrid", "dtype": args.dtype,
                               "tolerance": tolerance, "results": results}, indent=2) + "\n")
    print(json.dumps({"event": "passed", "path": str(path)}), flush=True)


if __name__ == "__main__":
    main()
