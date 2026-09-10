"""Pure linear numerator using FLA Ascend tiles and torch prefix states.

FLA 0.5.2's generic recurrent chunk_h kernel fails on the tested 910B stack
(unwritten intermediate states in FP32; compilation failure in BF16). This
project-local path computes forward/reverse chunk states with batched GEMMs
and FP32 cumsums, retaining FLA's tiled output/gradient kernels. It does not
patch installed packages, add a gate, or alter normalization.
"""
import torch
from torch.nn import functional as F
import fla  # Import before the Ascend compiler modifies triton.autotune.
from fla.ops.common.chunk_o import chunk_fwd_o, chunk_bwd_dqkwg, chunk_bwd_dv


def _blocks(x):
    b, t, h, d = x.shape
    return F.pad(x.transpose(1, 2).float(), (0, 0, 0, (-t) % 64)).reshape(b, h, -1, 64, d)


def _forward_states(k, v, h0):
    kc, vc = _blocks(k), _blocks(v)
    with torch.autocast(device_type="npu", enabled=False):
        cumulative = (kc.transpose(-1, -2) @ vc).cumsum(2)
    prior = torch.cat([torch.zeros_like(cumulative[:, :, :1]), cumulative[:, :, :-1]], dim=2)
    final = cumulative[:, :, -1]
    if h0 is not None:
        prior = prior + h0.float()[:, :, None]
        final = final + h0.float()
    return prior.transpose(1, 2).contiguous(), final


class _AscendLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, h0):
        ctx.set_materialize_grads(False)
        h, final = _forward_states(k, v, h0)
        out = chunk_fwd_o(q=q, k=k, v=v, h=h.to(q.dtype), scale=1.)
        ctx.save_for_backward(q, k, v, h0)
        return out, final

    @staticmethod
    def backward(ctx, do, dht):
        q, k, v, h0 = ctx.saved_tensors
        if do is None:
            do = torch.zeros_like(v)
        do = do.contiguous()
        h, _ = _forward_states(k, v, h0)
        qc, dc = _blocks(q), _blocks(do)
        with torch.autocast(device_type="npu", enabled=False):
            contributions = qc.transpose(-1, -2) @ dc
        reverse = contributions.flip(2).cumsum(2).flip(2)
        dh = torch.cat([reverse[:, :, 1:], torch.zeros_like(reverse[:, :, :1])], dim=2)
        dh0 = reverse[:, :, 0]
        if dht is not None:
            dh = dh + dht.float()[:, :, None]
            dh0 = dh0 + dht.float()
        dh = dh.transpose(1, 2).contiguous()
        dq, dk, _, _ = chunk_bwd_dqkwg(q=q, k=k, v=v, do=do, h=h, dh=dh, scale=1.)
        dv = chunk_bwd_dv(q=q, k=k, do=do, dh=dh, scale=1.)
        return dq, dk, dv, None if h0 is None else dh0.to(h0.dtype)


def ascend_linear_numerator(q, k, v, initial_state=None):
    """Equal-length [B,T,H,D], same Q/K/V head count, inclusive causal sum."""
    if q.device.type != "npu":
        raise ValueError("This kernel is only for Ascend NPU")
    if q.shape != k.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError("Expected matching batch, time and heads for Q/K/V")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Q/K/V must share compute dtype")
    return _AscendLinear.apply(q.contiguous(), k.contiguous(), v.contiguous(), initial_state)
