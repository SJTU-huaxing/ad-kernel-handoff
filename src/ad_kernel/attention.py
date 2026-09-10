"""Normalized positive linear attention with reversible per-feature scaling.

Inputs use [B,H,T,M] log features and [B,H,T,V] values. No forget gate,
amplitude clamp, epsilon kernel, or detached recurrent state is used.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

_numerical_splits = 0


def numerical_split_count():
    """Process-local count of exact-state numerical subdivisions."""
    return _numerical_splits


@dataclass
class LinearState:
    kv: torch.Tensor
    z: torch.Tensor
    log_scale: torch.Tensor


def dense_attention(logq, logk, v, valid=None):
    aq, bk = logq.amax(-1, keepdim=True), logk.amax(-1, keepdim=True)
    product = (logq-aq).exp() @ (logk-bk).exp().transpose(-1, -2)
    logits = product.log() + aq + bk.transpose(-1, -2)
    length = logq.shape[-2]
    mask = torch.ones(length, length, device=v.device, dtype=torch.bool).tril()
    if valid is not None:
        mask = mask & valid[:, None, None, :]
    logits = logits.masked_fill(~mask, -torch.inf)
    # Padding queries have no semantic output. Avoid all-masked softmax NaNs.
    if valid is not None:
        logits = torch.where(valid[:, None, :, None], logits, torch.zeros_like(logits))
    output = logits.softmax(-1) @ v
    return output if valid is None else output * valid[:, None, :, None]


def recurrent_attention(logq, logk, v, initial_state=None, valid=None):
    b, h, t, m = logq.shape
    if initial_state is None:
        kv = v.new_zeros(b, h, m, v.shape[-1])
        z = logk.new_zeros(b, h, m)
        g = logk.new_full((b, h, m), -torch.inf)
    else:
        kv, z, g = initial_state.kv, initial_state.z, initial_state.log_scale
    out = []
    for i in range(t):
        good = torch.ones(b, 1, 1, dtype=torch.bool, device=v.device) if valid is None else valid[:, i, None, None]
        lk = logk[:, :, i].masked_fill(~good, -torch.inf)
        new_g = torch.maximum(g, lk)
        safe_g = torch.where(torch.isfinite(new_g), new_g, torch.zeros_like(new_g))
        rescale = (g-safe_g).exp()
        phi_k = (lk-safe_g).exp()
        kv = kv * rescale[..., None] + phi_k[..., None] * v[:, :, i, None, :]
        z = z * rescale + phi_k
        read = logq[:, :, i] + safe_g
        phi_q = (read-read.amax(-1, keepdim=True)).exp()
        denom = (phi_q*z).sum(-1, keepdim=True)
        safe_denom = torch.where(good, denom, torch.ones_like(denom))
        output = (phi_q[..., None]*kv).sum(-2) / safe_denom
        out.append(torch.where(good, output, torch.zeros_like(output)))
        g = new_g
    return torch.stack(out, dim=2), LinearState(kv, z, g)


def _scaled_features(logq, logk, valid, initial_state):
    if valid is not None:
        logk = logk.masked_fill(~valid[:, None, :, None], -torch.inf)
    g = logk.amax(-2)
    if initial_state is not None:
        g = torch.maximum(g, initial_state.log_scale)
    g = torch.where(torch.isfinite(g), g, torch.zeros_like(g))
    # The choice of g is a gauge: the exact normalized output is independent
    # of it. Leave it differentiable; gradients are validated against dense.
    k = (logk-g[:, :, None, :]).exp()
    read = logq + g[:, :, None, :]
    q = (read-read.amax(-1, keepdim=True)).exp()
    if valid is not None:
        q = q * valid[:, None, :, None]
    kv0 = z0 = None
    if initial_state is not None:
        factor = (initial_state.log_scale-g).exp()
        kv0 = initial_state.kv * factor[..., None]
        z0 = initial_state.z * factor
    return q, k, g, kv0, z0


def chunk_attention(logq, logk, v, *, chunk_size=64, initial_state=None, valid=None,
                    backend="torch", compute_dtype=None):
    """O(T M V + T C (M+V)) exact reference; optional FLA numerator.

    Each batch row represents one document; use separate calls for document
    boundaries. ``valid`` masks padding, and state can continue the same doc.
    FLA's own normalize=True is intentionally not used (it adds epsilon).
    A future key can make globally scaled early-prefix denominators subnormal.
    In that case subdivide the execution span and carry the complete state and
    its gradient. The kernel, context, and denominator are never clipped.
    """
    q, k, g, kv0, z0 = _scaled_features(logq, logk, valid, initial_state)
    b, h, t, m = q.shape
    cumulative_k = k.cumsum(2)
    if z0 is not None:
        cumulative_k = cumulative_k + z0[:, :, None]
    denominator = (q * cumulative_k).sum(-1, keepdim=True)
    if valid is not None:
        denominator = torch.where(valid[:, None, :, None], denominator, torch.ones_like(denominator))
    # This threshold selects an algebraically equivalent execution path. It
    # is NOT added to or substituted for any attention denominator. The square
    # root leaves exponent headroom for backward intermediates in the dtype.
    if t > 1 and denominator.detach().amin().item() < math.sqrt(torch.finfo(q.dtype).tiny):
        global _numerical_splits
        _numerical_splits += 1
        cut = t // 2
        if t >= 2 * chunk_size:
            cut = max(chunk_size, (cut // chunk_size) * chunk_size)
        first, state = chunk_attention(logq[:, :, :cut], logk[:, :, :cut], v[:, :, :cut],
                                        chunk_size=chunk_size, initial_state=initial_state,
                                        valid=None if valid is None else valid[:, :cut],
                                        backend=backend, compute_dtype=compute_dtype)
        second, state = chunk_attention(logq[:, :, cut:], logk[:, :, cut:], v[:, :, cut:],
                                         chunk_size=chunk_size, initial_state=state,
                                         valid=None if valid is None else valid[:, cut:],
                                         backend=backend, compute_dtype=compute_dtype)
        return torch.cat([first, second], dim=2), state
    if backend in {"fla", "ascend-hybrid"}:
        import fla  # Must precede any custom Triton compilation in this process.
        if backend == "fla" and v.device.type == "npu":
            raise RuntimeError("Installed FLA pure-linear chunk states failed NPU correctness checks; use ascend-hybrid or torch")
        dtype = compute_dtype or v.dtype
        q, k, vc = q.to(dtype), k.to(dtype), v.to(dtype)
        if backend == "ascend-hybrid":
            from .ascend_linear import ascend_linear_numerator
            numerator, kv_final = ascend_linear_numerator(
                q.transpose(1, 2), k.transpose(1, 2), vc.transpose(1, 2), initial_state=kv0)
        else:
            from fla.ops.linear_attn import chunk_linear_attn
            numerator, kv_final = chunk_linear_attn(
                q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
                vc.transpose(1, 2).contiguous(), scale=1.0, normalize=False,
                initial_state=kv0, output_final_state=True,
            )
        numerator = numerator.transpose(1, 2).float()
        q, k = q.float(), k.float()
    elif backend == "torch":
        pad = (-t) % chunk_size
        qp = F.pad(q, (0, 0, 0, pad)).reshape(b, h, -1, chunk_size, m)
        kp = F.pad(k, (0, 0, 0, pad)).reshape(b, h, -1, chunk_size, m)
        vp = F.pad(v, (0, 0, 0, pad)).reshape(b, h, -1, chunk_size, v.shape[-1])
        contributions = kp.transpose(-1, -2) @ vp
        cumulative = contributions.cumsum(2)
        prior = torch.cat([torch.zeros_like(cumulative[:, :, :1]), cumulative[:, :, :-1]], dim=2)
        if kv0 is not None:
            prior = prior + kv0[:, :, None]
        intra = (qp @ kp.transpose(-1, -2)).tril() @ vp
        numerator = (qp @ prior + intra).reshape(b, h, -1, v.shape[-1])[:, :, :t]
        kv_final = cumulative[:, :, -1] + (kv0 if kv0 is not None else 0)
    else:
        raise ValueError(backend)
    if backend != "torch":
        # Those backends can quantize Q/K before their numerator kernels.
        # Preserve the original matching denominator precision convention.
        cumulative_k = k.cumsum(2)
        if z0 is not None:
            cumulative_k = cumulative_k + z0[:, :, None]
        denominator = (q * cumulative_k).sum(-1, keepdim=True)
        if valid is not None:
            denominator = torch.where(valid[:, None, :, None], denominator, torch.ones_like(denominator))
    output = numerator / denominator
    if valid is not None:
        output = output * valid[:, None, :, None]
    final_z = cumulative_k[:, :, -1]
    # An all-padding span has no key mass. Its safe temporary zero scale must
    # not suppress a subsequent valid key with a large negative log amplitude.
    final_g = torch.where(final_z > 0, g, torch.full_like(g, -torch.inf))
    return output, LinearState(kv_final, final_z, final_g)
