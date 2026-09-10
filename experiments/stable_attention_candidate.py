"""Prototype: split numerically unsafe spans, carrying the exact causal state."""
import math
import torch
from ad_kernel.attention import chunk_attention, _scaled_features


def stable_chunk(logq, logk, v, *, chunk_size=64, initial_state=None, valid=None, backend="torch", compute_dtype=None, stats=None):
    q, k, _, _, z0 = _scaled_features(logq, logk, valid, initial_state)
    z = k.cumsum(2)
    if z0 is not None:
        z = z + z0[:, :, None]
    denominator = (q * z).sum(-1)
    if valid is not None:
        denominator = torch.where(valid[:, None], denominator, torch.ones_like(denominator))
    length = logq.shape[2]
    threshold = math.sqrt(torch.finfo(q.dtype).tiny)
    if length > 1 and denominator.detach().amin().item() < threshold:
        if stats is not None:
            stats.append({"length": length, "minimum_denominator": denominator.detach().amin().item()})
        split = length // 2
        if length >= 2 * chunk_size:
            split = max(chunk_size, (split // chunk_size) * chunk_size)
        a, state = stable_chunk(logq[:, :, :split], logk[:, :, :split], v[:, :, :split],
                                chunk_size=chunk_size, initial_state=initial_state,
                                valid=None if valid is None else valid[:, :split], backend=backend,
                                compute_dtype=compute_dtype, stats=stats)
        b, state = stable_chunk(logq[:, :, split:], logk[:, :, split:], v[:, :, split:],
                                chunk_size=chunk_size, initial_state=state,
                                valid=None if valid is None else valid[:, split:], backend=backend,
                                compute_dtype=compute_dtype, stats=stats)
        return torch.cat([a, b], 2), state
    return chunk_attention(logq, logk, v, chunk_size=chunk_size, initial_state=initial_state,
                           valid=valid, backend=backend, compute_dtype=compute_dtype)
