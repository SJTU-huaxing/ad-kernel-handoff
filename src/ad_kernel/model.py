"""Conventional decoder backbone for controlled all-layer attention studies."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from .features import ADFeatures
from .baselines import HedgehogFeatures, FavorFeatures
from .attention import chunk_attention


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    width: int = 768
    layers: int = 12
    heads: int = 12
    intermediate: int = 2048
    method: str = "ad"
    feature_dim: int = 64
    feature_hidden: int = 96
    feature_seed: int = 11
    rope_theta: float = 10000.
    norm_eps: float = 1e-6
    backend: str = "ascend-hybrid"
    attention_dtype: str = "float32"
    fused_loss: bool = True


def rotate(x, cos, sin):
    first, second = x.float().chunk(2, dim=-1)
    return x.float() * cos + torch.cat([-second, first], dim=-1) * sin


class TraditionalFeatures(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method

    def log_feature(self, x, side):
        if self.method == "elu":
            # log(ELU(x)+1), stable also for large negative inputs.
            return torch.where(x > 0, torch.log1p(F.relu(x)), x)
        if self.method == "softplus":
            return torch.where(x < -20, x, F.softplus(x).log())
        raise ValueError(self.method)


class Attention(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.config = config
        self.dim = config.width // config.heads
        if config.width % config.heads or self.dim % 2:
            raise ValueError("Head dimension must be an even divisor of model width")
        self.qkv = nn.Linear(config.width, config.width * 3, bias=False)
        self.out = nn.Linear(config.width, config.width, bias=False)
        # Feature initialization must not perturb the shared backbone RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.feature_seed + 1009 * layer)
            if config.method in {"ad", "exp"}:
                self.features = ADFeatures(heads=config.heads, dim=self.dim, hidden=config.feature_hidden,
                                           m=config.feature_dim, kind=config.method)
            elif config.method in {"hh_exp", "hh_softmax"}:
                self.features = HedgehogFeatures(heads=config.heads, dim=self.dim, m=config.feature_dim,
                                                softmax=config.method == "hh_softmax", bias=True)
            elif config.method == "favor":
                self.features = FavorFeatures(heads=config.heads, dim=self.dim, m=config.feature_dim, scale_inputs=False)
            elif config.method in {"elu", "softplus"}:
                if config.feature_dim != self.dim:
                    raise ValueError("Elementwise maps retain the original head dimension")
                self.features = TraditionalFeatures(config.method)
            elif config.method == "softmax":
                self.features = None
            else:
                raise ValueError(config.method)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.config.heads, self.dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        if self.features is None:
            result = F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v, is_causal=True)
        else:
            # All kernels use the same softmax temperature convention; Q/K
            # each receive d^(-1/4), including the feature-map controls.
            with torch.autocast(device_type=x.device.type, enabled=False):
                q, k = q * self.dim ** (-.25), k * self.dim ** (-.25)
                lq, lk = self.features.log_feature(q, "q"), self.features.log_feature(k, "k")
                result, _ = chunk_attention(lq, lk, v.float(), backend=self.config.backend,
                                           compute_dtype=getattr(torch, self.config.attention_dtype))
            result = result.to(v.dtype)
        return self.out(result.transpose(1, 2).reshape(b, t, -1))


class Block(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.width, eps=config.norm_eps)
        self.attention = Attention(config, layer)
        self.mlp_norm = nn.RMSNorm(config.width, eps=config.norm_eps)
        self.up = nn.Linear(config.width, 2 * config.intermediate, bias=False)
        self.down = nn.Linear(config.intermediate, config.width, bias=False)

    def forward(self, x, cos, sin):
        x = x + self.attention(self.attention_norm(x), cos, sin)
        gate, value = self.up(self.mlp_norm(x)).chunk(2, -1)
        return x + self.down(F.silu(gate) * value)


class LinearLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.layers)])
        self.final_norm = nn.RMSNorm(config.width, eps=config.norm_eps)
        head_dim = config.width // config.heads
        self.register_buffer("inverse_frequency", config.rope_theta ** (-torch.arange(0, head_dim, 2).float() / head_dim), persistent=False)
        self.apply(self._initialize)
        for block in self.blocks:
            nn.init.normal_(block.attention.out.weight, std=.02 / math.sqrt(2 * config.layers))
            nn.init.normal_(block.down.weight, std=.02 / math.sqrt(2 * config.layers))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)

    def forward(self, tokens, labels=None):
        x = self.embedding(tokens)
        angles = torch.arange(tokens.shape[1], device=tokens.device).float()[:, None] * self.inverse_frequency
        phase = torch.cat([angles, angles], -1)[None, None]
        cos, sin = phase.cos(), phase.sin()
        for block in self.blocks:
            x = block(x, cos, sin)
        hidden = self.final_norm(x)
        if labels is None:
            return F.linear(hidden, self.embedding.weight)
        if not torch.is_grad_enabled():
            flat, targets = hidden.flatten(0, 1), labels.flatten()
            total = hidden.new_zeros((), dtype=torch.float32)
            for begin in range(0, len(flat), 256):
                logits = F.linear(flat[begin:begin + 256], self.embedding.weight).float()
                total = total + F.cross_entropy(logits, targets[begin:begin + 256], reduction="sum")
            return total / targets.ne(-100).sum()
        if self.config.fused_loss and tokens.device.type == "npu":
            from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
            return FusedLinearCrossEntropyLoss(num_chunks=8, reduction="mean")(hidden.contiguous(), labels.contiguous(), self.embedding.weight)
        return F.cross_entropy(F.linear(hidden, self.embedding.weight).float().flatten(0, 1), labels.flatten())

    def parameter_budget(self):
        total = sum(p.numel() for p in self.parameters())
        features = sum(p.numel() for name, p in self.named_parameters() if ".features." in name)
        dim = self.config.width // self.config.heads
        return {"total_parameters": total, "backbone_parameters": total - features,
                "feature_parameters": features, "feature_dim": self.config.feature_dim,
                "linear_state_scalars_per_sequence": None if self.config.method == "softmax" else
                    self.config.layers * self.config.heads * self.config.feature_dim * (dim + 2),
                "state_includes": "S, z and the per-feature numerical scale; excludes training activations",
                "config": asdict(self.config)}
