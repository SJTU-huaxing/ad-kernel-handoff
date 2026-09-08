import copy
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

P = Path(__file__).resolve().parent
ROOT = P.parent
OLD = ROOT / 'hedgehog_matched'
sys.path.insert(0, str(OLD))
import models as parent
import train_product as product

source = parent.source
H, D, HEADS, KI = parent.H, parent.D, parent.HEADS, parent.KI
SEEDS = [11, 29, 47]
REGIMES = ['product_i', 'causal_i', 'causal_kl']
VARIANTS = ['reduced_plain', 'reduced_matched']
for folder in ['fits', 'results', 'checks', 'logs', 'figures']:
    (P / folder).mkdir(parents=True, exist_ok=True)


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))


class Reduced(nn.Module):
    def __init__(self, norm, variant, regime, numerical_scale):
        super().__init__()
        self.variant, self.regime = variant, regime
        self.m, self.heads = 64, norm['q_mean'].shape[0]
        self.runtime_raw = False
        template = parent.Matched(norm, 'ad_raw')
        self.qnet = copy.deepcopy(template.qnet)
        self.qnet.w2 = nn.Parameter(template.qnet.w2[:, :63].detach().clone().contiguous())
        self.knet = copy.deepcopy(template.knet)
        for key, value in norm.items():
            self.register_buffer(key, value)
        self.register_buffer('numerical_scale', numerical_scale.detach().clone())
        matched = variant == 'reduced_matched'
        self.q_hidden_bias = nn.Parameter(torch.zeros(self.heads, 192)) if matched else None
        self.q_direction_bias = (nn.Parameter(torch.zeros(self.heads))
                                 if matched and regime != 'causal_kl' else None)
        expected = 73536 if not matched else (73728 if regime == 'causal_kl' else 73729)
        assert sum(p.numel() for p in self.parameters()) // self.heads == expected

    def raw_log_feature(self, x, side):
        x = (x - getattr(self, side + '_mean')[:, None]) / getattr(self, side + '_std')[:, None]
        if side == 'q':
            hidden = x @ self.qnet.w1.transpose(-1, -2)
            if self.q_hidden_bias is not None:
                hidden = hidden + self.q_hidden_bias[:, None]
            z = F.silu(hidden) @ self.qnet.w2.transpose(-1, -2)
            if self.q_direction_bias is not None:
                z = torch.cat([z[..., :1] + self.q_direction_bias[:, None, None], z[..., 1:]], -1)
            return torch.cat([z, torch.zeros_like(z[..., :1])], -1).log_softmax(-1)
        z = self.knet(x)
        direction = torch.cat([z[..., :63], torch.zeros_like(z[..., -1:])], -1)
        return direction.log_softmax(-1) + z[..., -1:]

    def log_feature(self, x, side):
        answer = self.raw_log_feature(x, side)
        # Only a numerical representation of h/exp(c). raw_log_feature defines h.
        return answer if self.runtime_raw else answer - self.numerical_scale[:, None, None] / 2

    log_matrix = parent.source.OriginalPair.log_matrix


def load_fit(name, dtype=torch.float64, device='cuda'):
    box = torch.load(P / 'fits' / (name + '.pt'), weights_only=True)
    sd, meta = box['state_dict'], box['metadata']
    norm = {key: sd[key] for key in ['q_mean', 'q_std', 'k_mean', 'k_std']}
    net = Reduced(norm, meta['variant'], meta['regime'], sd['numerical_scale']).to(dtype=dtype, device=device)
    net.load_state_dict(sd)
    return net.eval(), meta


def inference_cancelled(original):
    norm = {key: getattr(original, key) for key in ['q_mean', 'q_std', 'k_mean', 'k_std']}
    net = Reduced(norm, 'reduced_plain', 'causal_kl', torch.zeros(original.heads, device=original.q_mean.device))
    net = net.to(device=original.q_mean.device, dtype=original.q_mean.dtype)
    with torch.no_grad():
        net.qnet.w1.copy_(original.qnet.w1)
        net.qnet.w2.copy_(original.qnet.w2[:, :63])
        net.knet.load_state_dict(original.knet.state_dict())
    net.runtime_raw = True
    return net.eval()


def plan():
    return json.loads((P / 'results' / 'plan.json').read_text())


def old_name(regime, kind, seed):
    prefix = 'product_' if regime == 'product_i' else ''
    suffix = 'kl' if regime == 'causal_kl' else 'raw'
    return f'{prefix}{kind}_{suffix}_s{seed}_lr0.002'
