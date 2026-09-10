"""Device-independent, checkpoint-compatible features.

Historical tensors are [head, token, dimension]. Leading batch dimensions are
also supported: [batch, head, token, dimension]. No query amplitude, external C,
clipping, gates or extra feature factors are introduced.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class HeadMLP(nn.Module):
    def __init__(self, heads: int, dim: int, hidden: int, outputs: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(heads, hidden, dim) / math.sqrt(dim))
        self.w2 = nn.Parameter(torch.randn(heads, outputs, hidden) / math.sqrt(hidden) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x @ self.w1.transpose(-1, -2)) @ self.w2.transpose(-1, -2)


class ADFeatures(nn.Module):
    def __init__(self, heads=24, dim=128, hidden=192, m=64, *, kind="ad",
                 initialization="standard", norm=None, numerical_scale=None):
        super().__init__()
        if kind not in {"ad", "exp", "direction_only"}:
            raise ValueError(kind)
        if initialization not in {"standard", "matched_zero"}:
            raise ValueError(initialization)
        self.heads, self.dim, self.hidden, self.m = heads, dim, hidden, m
        self.kind, self.initialization = kind, initialization
        self.runtime_raw = True
        # Preserve the original full-m query initialization RNG consumption.
        self.qnet = HeadMLP(heads, dim, hidden, m)
        self.knet = HeadMLP(heads, dim, hidden, m)
        self.qnet.w2 = nn.Parameter(self.qnet.w2[:, :m-1].detach().clone().contiguous())
        for side in ["q", "k"]:
            for stat in ["mean", "std"]:
                key = f"{side}_{stat}"
                value = (norm[key].detach().clone() if norm is not None else
                         torch.zeros(heads, dim) if stat == "mean" else torch.ones(heads, dim))
                self.register_buffer(key, value)
        self.register_buffer("numerical_scale", numerical_scale.detach().clone() if numerical_scale is not None
                             else torch.zeros(heads))
        if initialization == "matched_zero":
            with torch.no_grad():
                self.knet.w2.zero_()

    def raw_log_feature(self, x, side):
        if side not in {"q", "k"}:
            raise ValueError(side)
        x = (x - getattr(self, side + "_mean")[:, None, :]) / getattr(self, side + "_std")[:, None, :]
        z = getattr(self, side + "net")(x)
        if side == "k" and self.kind == "exp":
            return z
        direction = z if side == "q" else z[..., :-1]
        log_direction = torch.cat([direction, torch.zeros_like(z[..., :1])], -1).log_softmax(-1)
        if side == "q" or self.kind == "direction_only":
            return log_direction
        return log_direction + z[..., -1:]

    def log_feature(self, x, side):
        result = self.raw_log_feature(x, side)
        return result if self.runtime_raw else result - self.numerical_scale[:, None, None] / 2

    def log_matrix(self, q, k):
        a, b = self.log_feature(q, "q"), self.log_feature(k, "k")
        aq, bk = a.amax(-1, keepdim=True), b.amax(-1, keepdim=True)
        product = (a - aq).exp() @ (b - bk).exp().transpose(-1, -2)
        return product.clamp_min(torch.finfo(product.dtype).tiny).log() + aq + bk.transpose(-1, -2)

    @property
    def parameters_per_head(self):
        return sum(p.numel() for p in self.parameters()) // self.heads


def load_historical_features(path: str | Path, *, device="cpu", dtype=torch.float64):
    box = torch.load(path, map_location="cpu", weights_only=True)
    sd, meta = box["state_dict"], box["metadata"]
    required = {"qnet.w1", "qnet.w2", "knet.w1", "knet.w2", "q_mean", "q_std", "k_mean", "k_std", "numerical_scale"}
    if set(sd) != required:
        raise ValueError(f"Expected pure AD / matched EXP checkpoint; unexpected fields: {set(sd) ^ required}")
    heads, hidden, dim = sd["qnet.w1"].shape
    m = sd["knet.w2"].shape[1]
    if sd["qnet.w2"].shape != (heads, m-1, hidden):
        raise ValueError("Query output layout differs from the handed-off pure AD model")
    net = ADFeatures(heads, dim, hidden, m, kind=meta.get("map_kind", "ad"),
                     initialization=meta.get("initialization", "standard"),
                     norm={key: sd[key] for key in ["q_mean", "q_std", "k_mean", "k_std"]},
                     numerical_scale=sd["numerical_scale"])
    net.load_state_dict(sd, strict=True)
    return net.to(device=device, dtype=dtype), meta

