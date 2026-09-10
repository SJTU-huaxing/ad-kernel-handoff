"""Independent Q/K Hedgehog and frozen orthogonal positive random features.

Historical adapters preserve their original normalization and state layout.
Learned feature parameters and fixed random buffers are reported separately.
"""
import math
import torch
from torch import nn
from .features import ADFeatures


class HedgehogFeatures(nn.Module):
    def __init__(self, heads=24, dim=128, m=576, *, softmax=False, norm=None, bias=False):
        super().__init__()
        if m % 2:
            raise ValueError("Hedgehog concatenates positive and negative branches")
        self.heads, self.dim, self.m = heads, dim, m
        self.softmax = softmax
        width = m // 2
        initial = [torch.eye(dim)[None].repeat(heads, 1, 1)[:, :width]]
        remaining = max(0, width - dim)
        while remaining:
            z = torch.randn(heads, dim, dim)
            u, r = torch.linalg.qr(z)
            u = u * r.diagonal(dim1=-2, dim2=-1).sign()[:, None]
            initial.append(u[:, :min(dim, remaining)])
            remaining -= min(dim, remaining)
        initial = torch.cat(initial, 1)
        self.qweight = nn.Parameter(initial.clone())
        self.kweight = nn.Parameter(initial.clone())
        if bias:
            self.qbias = nn.Parameter(torch.zeros(heads, width))
            self.kbias = nn.Parameter(torch.zeros(heads, width))
        else:
            self.qbias = self.kbias = None
        for side in ["q", "k"]:
            for stat in ["mean", "std"]:
                key = side + "_" + stat
                value = norm[key].detach().clone() if norm is not None else (
                    torch.zeros(heads, dim) if stat == "mean" else torch.ones(heads, dim))
                self.register_buffer(key, value)
        self.register_buffer("log_scale", torch.zeros(heads))

    def log_feature(self, x, side):
        x = (x - getattr(self, side + "_mean")[:, None]) / getattr(self, side + "_std")[:, None]
        z = x @ getattr(self, side + "weight").transpose(-1, -2)
        bias = getattr(self, side + "bias")
        if bias is not None:
            z = z + bias[:, None]
        answer = (torch.cat([z.log_softmax(-1), (-z).log_softmax(-1)], -1) if self.softmax else
                  torch.cat([z, -z], -1) - .5 * math.log(self.m))
        return answer + self.log_scale[:, None, None] / 2

    raw_log_feature = log_feature
    log_matrix = ADFeatures.log_matrix

    @property
    def parameters_per_head(self):
        return sum(p.numel() for p in self.parameters()) // self.heads


class FavorFeatures(nn.Module):
    def __init__(self, heads=24, dim=128, m=64, *, norm=None, scale_inputs=True):
        super().__init__()
        self.heads, self.dim, self.m = heads, dim, m
        self.scale_inputs = scale_inputs
        blocks = []
        remaining = m
        while remaining:
            g = torch.randn(heads, dim, dim)
            u, r = torch.linalg.qr(g)
            u = u * r.diagonal(dim1=-2, dim2=-1).sign()[:, None]
            radii = torch.randn(heads, dim, dim).norm(dim=-1)
            take = min(dim, remaining)
            blocks.append(u.transpose(-1, -2)[:, :take] * radii[:, :take, None])
            remaining -= take
        self.register_buffer("weight", torch.cat(blocks, 1))
        self.register_buffer("bias", torch.zeros(heads, m))
        for side in ["q", "k"]:
            for stat in ["mean", "std"]:
                key = side + "_" + stat
                value = norm[key].detach().clone() if norm is not None else (
                    torch.zeros(heads, dim) if stat == "mean" else torch.ones(heads, dim))
                self.register_buffer(key, value)

    def log_feature(self, x, side):
        # Favor's historical source stores norm buffers but does not use them.
        z = x * self.dim ** (-.25) if self.scale_inputs else x
        return (z @ self.weight.transpose(-1, -2) - .5 * z.square().sum(-1, keepdim=True)
                + self.bias[:, None] / 2 - .5 * math.log(self.m))

    raw_log_feature = log_feature
    log_matrix = ADFeatures.log_matrix

    @property
    def parameters_per_head(self):
        return 0
