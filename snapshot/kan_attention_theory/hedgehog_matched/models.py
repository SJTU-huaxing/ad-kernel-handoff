import json
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

P = Path(__file__).resolve().parent
ROOT = P.parent
sys.path.insert(0, str(ROOT / 'causal_direction'))
import common as source

H, D, HEADS, KI = source.H, source.D, source.HEADS, source.KI
KINDS = ['ad_raw', 'hh_raw', 'ad_kl', 'hh_kl', 'hh_softmax_kl']
SEEDS = [11, 29, 47]
for folder in ['fits', 'results', 'checks', 'figures']:
    (P / folder).mkdir(exist_ok=True)


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))


class ADNetwork(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(heads, 192, 128) / math.sqrt(128))
        self.w2 = nn.Parameter(torch.randn(heads, 64, 192) / math.sqrt(192) * .1)

    def forward(self, x):
        return F.silu(x @ self.w1.transpose(-1, -2)) @ self.w2.transpose(-1, -2)


class Matched(nn.Module):
    def __init__(self, norm, kind):
        super().__init__()
        self.kind = kind
        self.heads = norm['q_mean'].shape[0]
        self.m = 64 if kind.startswith('ad') else 576
        for key, value in norm.items():
            self.register_buffer(key, value)
        if kind.endswith('_raw'):
            self.log_scale = nn.Parameter(torch.zeros(self.heads))
        else:
            self.register_buffer('log_scale', torch.zeros(self.heads))
        if kind.startswith('ad'):
            self.qnet = ADNetwork(self.heads)
            self.knet = ADNetwork(self.heads)
        else:
            blocks = [torch.eye(128)[None].repeat(self.heads, 1, 1)]
            for size in [128, 32]:
                z = torch.randn(self.heads, 128, 128)
                u, r = torch.linalg.qr(z)
                u = u * r.diagonal(dim1=-2, dim2=-1).sign()[:, None]
                blocks.append(u[:, :size])
            initial = torch.cat(blocks, 1)
            self.qweight = nn.Parameter(initial.clone())
            self.kweight = nn.Parameter(initial.clone())

    def log_feature(self, x, side):
        x = (x - getattr(self, side + '_mean')[:, None]) / getattr(self, side + '_std')[:, None]
        if self.kind.startswith('ad'):
            z = getattr(self, side + 'net')(x)
            shape = torch.cat([z[..., :-1], torch.zeros_like(z[..., -1:])], -1)
            answer = shape.log_softmax(-1) + z[..., -1:] + .5 * math.log(self.m)
        else:
            z = x @ getattr(self, side + 'weight').transpose(-1, -2)
            if self.kind == 'hh_softmax_kl':
                answer = torch.cat([z.log_softmax(-1), (-z).log_softmax(-1)], -1)
            else:
                answer = torch.cat([z, -z], -1) - .5 * math.log(self.m)
        return answer + self.log_scale[:, None, None] / 2

    log_matrix = source.OriginalPair.log_matrix


def load_fit(name, dtype=torch.float64, device='cuda'):
    box = torch.load(P / 'fits' / (name + '.pt'), weights_only=True)
    sd = box['state_dict']
    norm = {key: sd[key] for key in ['q_mean', 'q_std', 'k_mean', 'k_std']}
    net = Matched(norm, box['metadata']['kind']).to(dtype=dtype, device=device)
    net.load_state_dict(sd)
    return net.eval(), box['metadata']


@torch.inference_mode()
def evaluate(net, ds, scale):
    rows = []
    index = KI.to(next(net.parameters()).device)
    scale = scale.to(index.device).double()
    for i in range(len(ds['q'])):
        q = ds['q'][i].to(index.device).double()
        k = ds['k'][i].to(index.device).double()[index]
        v = ds['v'][i].to(index.device).double()[index]
        pos = ds['query_positions'][i].to(index.device)
        mask = (torch.arange(k.shape[1], device=index.device)[None] <= pos[:, None])[None]
        lt = q @ k.transpose(-1, -2) / math.sqrt(D) - scale[:, None, None]
        lp = net.log_matrix(q, k)
        balanced, kl, mass = source.rows_loss(lp, lt, mask, 1.)
        t = lt.masked_fill(~mask, -torch.inf)
        p = lp.masked_fill(~mask, -torch.inf)
        a, b = t.softmax(-1), p.softmax(-1)
        y, yh = a @ v, b @ v
        truth, pred = t.exp(), p.exp()
        log_residual = (lt - lp).masked_fill(~mask, 0)
        row_mass_log_error = p.logsumexp(-1) - t.logsumexp(-1)
        row = dict(
            ordinal=i, kl=kl.mean(-1).tolist(), mass=mass.mean(-1).tolist(),
            balanced=balanced.mean(-1).tolist(),
            raw_sse=(truth - pred).square().sum((-1, -2)).tolist(),
            raw_energy=truth.square().sum((-1, -2)).tolist(),
            raw_i=(pred - truth + truth * log_residual).sum((-1, -2)).tolist(),
            raw_truth_mass=truth.sum((-1, -2)).tolist(),
            log_mass_mae=row_mass_log_error.abs().mean(-1).tolist(),
            output_nmse=((y - yh).square().sum((-1, -2)) /
                         y.square().sum((-1, -2)).clamp_min(1e-30)).tolist())
        assert all(math.isfinite(v) for key, val in row.items() if key != 'ordinal' for v in val)
        rows.append(row)
    summary = {key: torch.tensor([r[key] for r in rows], dtype=torch.float64).mean(0).tolist()
               for key in rows[0] if key != 'ordinal'}
    for a, b, name in [('raw_sse', 'raw_energy', 'raw_nmse'),
                       ('raw_i', 'raw_truth_mass', 'relative_raw_i')]:
        summary[name] = (torch.tensor(summary[a], dtype=torch.float64) /
                         torch.tensor(summary[b], dtype=torch.float64)).tolist()
    return dict(summary=summary, documents=rows)
