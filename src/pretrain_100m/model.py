"""FLA-style ~100M decoder with controlled positive-kernel attention.

Architecture reference: fla-org/flame configs/transformer_340M.json.
Uses installed FLA RMSNorm, GatedMLP, HedgehogFeatureMap and fused linear CE.
The log-space Hedgehog adapter is algebraically the official feature map,
including its factor of two, joint softmax, biases and sharing across heads.
"""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
import fla  # Configure Ascend before any Triton compilation.
from fla.modules import RMSNorm, GatedMLP
from fla.modules.feature_map import HedgehogFeatureMap
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from ad_kernel.features import ADFeatures
from ad_kernel.baselines import FavorFeatures
from ad_kernel.attention import chunk_attention, _scaled_features, LinearState


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    hidden_size: int = 640
    num_hidden_layers: int = 12
    num_heads: int = 10
    intermediate_size: int = 2048
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.
    norm_eps: float = 1e-6
    initializer_range: float = .02
    tie_word_embeddings: bool = True
    method: str = "ad64"
    feature_hidden: int = 96
    feature_seed: int = 11
    attention_backend: str = "cann"
    chunk_size: int = 64
    fuse_norm: bool = True
    fuse_swiglu: bool = True
    fuse_linear_cross_entropy: bool = True
    loss_chunks: int = 8


class FLAHedgehogLogFeatures(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_map = HedgehogFeatureMap(dim)
        self.k_map = HedgehogFeatureMap(dim)

    def log_feature(self, x, side):
        # log(official_map(x)) without exp underflow followed by log(0).
        mapped = getattr(self, side + "_map").layer(x) * 2
        return F.log_softmax(torch.cat((mapped, -mapped), -1), -1)


def recurrent_fla(logq, logk, v, chunk_size):
    """Verified FLA recurrent numerator; the exact denominator has no epsilon.

    Unsafe global gauges use the existing differentiable CANN subdivision.
    This is explicitly an alternative backend, not the broken FLA chunk path.
    """
    from fla.ops.linear_attn import fused_recurrent_linear_attn
    q, k, g, _, _ = _scaled_features(logq, logk, None, None)
    kz = k.cumsum(2)
    denominator = (q * kz).sum(-1, keepdim=True)
    if denominator.detach().amin().item() < math.sqrt(torch.finfo(q.dtype).tiny):
        return chunk_attention(logq, logk, v, backend="torch", chunk_size=chunk_size)[0]
    result, _ = fused_recurrent_linear_attn(
        q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(), scale=1., normalize=False,
        output_final_state=False)
    return result.transpose(1, 2) / denominator


class Attention(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.config = config
        self.dim = config.hidden_size // config.num_heads
        assert config.hidden_size % config.num_heads == 0 and self.dim % 2 == 0
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # Keep the backbone RNG identical for all four methods.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.feature_seed + 1009 * layer)
            if config.method == "ad64":
                self.features = ADFeatures(config.num_heads, self.dim, config.feature_hidden, 64)
            elif config.method == "favor64":
                self.features = FavorFeatures(config.num_heads, self.dim, 64, scale_inputs=False)
            elif config.method == "hedgehog":
                self.features = FLAHedgehogLogFeatures(self.dim)
            elif config.method == "softmax":
                self.features = None
            else:
                raise ValueError(config.method)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.config.num_heads, self.dim).permute(2, 0, 3, 1, 4).unbind(0)
        def rotate(a):
            a = a.float()
            first, second = a.chunk(2, -1)
            return a * cos + torch.cat((-second, first), -1) * sin
        q, k = rotate(q), rotate(k)
        if self.features is None:
            result = F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v,
                                                   is_causal=True, dropout_p=0.)
        else:
            with torch.autocast(x.device.type, enabled=False):
                q, k = q * self.dim ** (-.25), k * self.dim ** (-.25)
                lq, lk = self.features.log_feature(q, "q"), self.features.log_feature(k, "k")
                if self.config.attention_backend == "fla_recurrent":
                    result = recurrent_fla(lq, lk, v.float(), self.config.chunk_size)
                elif self.config.attention_backend == "cann":
                    result, _ = chunk_attention(lq, lk, v.float(), backend="torch", chunk_size=self.config.chunk_size)
                else:
                    raise ValueError(self.config.attention_backend)
            result = result.to(v.dtype)
        return self.out(result.transpose(1, 2).reshape(b, t, -1))


class Block(nn.Module):
    def __init__(self, c, layer):
        super().__init__()
        norm = RMSNorm if c.fuse_norm else nn.RMSNorm
        self.attn_norm = norm(c.hidden_size, eps=c.norm_eps)
        self.attn = Attention(c, layer)
        self.mlp_norm = norm(c.hidden_size, eps=c.norm_eps)
        self.mlp = GatedMLP(c.hidden_size, intermediate_size=c.intermediate_size,
                            hidden_act="swish", fuse_swiglu=c.fuse_swiglu)
        self.fuse_norm = c.fuse_norm

    def forward(self, x, cos, sin):
        a = self.attn(self.attn_norm(x), cos, sin)
        if self.fuse_norm:
            h, residual = self.mlp_norm(a, x, True)
        else:
            residual = x + a
            h = self.mlp_norm(residual)
        if h.device.type == "cpu":
            mlp = self.mlp.down_proj(F.silu(self.mlp.gate_proj(h)) * self.mlp.up_proj(h))
        else:
            mlp = self.mlp(h)
        return residual + mlp


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.tie_word_embeddings, "This protocol fixes tied embeddings"
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config, i) for i in range(config.num_hidden_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.criterion = FusedLinearCrossEntropyLoss(num_chunks=config.loss_chunks, reduction="sum")
        d = config.hidden_size // config.num_heads
        inv = config.rope_theta ** (-torch.arange(0, d, 2).float() / d)
        self.register_buffer("inv_freq", inv, persistent=False)
        # Do NOT run apply(_init) through official Hedgehog: preserve identity.
        nn.init.normal_(self.embedding.weight, std=config.initializer_range)
        for layer in self.layers:
            for module in [layer.attn.qkv, layer.attn.out, layer.mlp.gate_proj,
                           layer.mlp.up_proj, layer.mlp.down_proj]:
                nn.init.normal_(module.weight, std=config.initializer_range)
            for module in [layer.attn.out, layer.mlp.down_proj]:
                nn.init.normal_(module.weight, std=config.initializer_range / math.sqrt(2 * config.num_hidden_layers))

    def hidden_states(self, tokens):
        assert tokens.ndim == 2 and tokens.shape[1] <= self.config.max_position_embeddings
        x = self.embedding(tokens)
        angles = torch.arange(tokens.shape[1], device=tokens.device).float()[:, None] * self.inv_freq
        phase = torch.cat((angles, angles), -1)[None, None]
        cos, sin = phase.cos(), phase.sin()
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)

    def forward(self, tokens, labels=None):
        hidden = self.hidden_states(tokens)
        if labels is None:
            return F.linear(hidden, self.embedding.weight)
        if self.config.fuse_linear_cross_entropy and tokens.device.type == "npu" and torch.is_grad_enabled():
            return self.criterion(hidden.contiguous(), labels.contiguous(), self.embedding.weight)
        # Bounded logits memory for evaluation and the CPU correctness oracle.
        flat, target = hidden.flatten(0, 1), labels.flatten()
        total = hidden.new_zeros((), dtype=torch.float32)
        for i in range(0, flat.shape[0], 256):
            logits = F.linear(flat[i:i+256], self.embedding.weight).float()
            total = total + F.cross_entropy(logits, target[i:i+256], reduction="sum", ignore_index=-100)
        return total

    def parameter_budget(self):
        total = sum(p.numel() for p in self.parameters())
        features = sum(p.numel() for n, p in self.named_parameters() if ".features." in n)
        return {"total": total, "backbone": total - features, "learned_features": features,
                "fixed_feature_buffers": sum(b.numel() for n, b in self.named_buffers() if ".features." in n),
                "config": asdict(self.config)}
