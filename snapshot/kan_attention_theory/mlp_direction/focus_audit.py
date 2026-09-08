"""Focused amplitude/direction audit; no training or new benchmark.

Reanalyse saved full-product measurements and check identities on a small,
deterministic subset of cached real Q/K. The subset is an algebra audit, not
a new generalisation test or a population-optimality certificate.
"""
import hashlib
import json
import math
from pathlib import Path

import torch

from core import Pair

P = Path(__file__).resolve().parent


def divergence(t, h):
    return t * (t.log() - h.log()) - t + h


def relative_max(x, y):
    return float((x - y).abs().amax() / y.abs().amax().clamp_min(1e-30))


def main():
    torch.set_num_threads(4)
    summary_path = P / 'results/product_summary.json'
    summary = json.loads(summary_path.read_text())
    reviewed = []
    for ds in ['fresh_wiki3', 'swde3']:
        rows = {r['method']: r for r in summary if r['dataset'] == ds}
        main_row = rows['factorized_both']
        reviewed.append(dict(
            dataset=ds,
            method='factorized_both',
            balanced_risk=main_row['query_balanced_divergence'],
            direction_risk=main_row['directional_kl'],
            amplitude_risk=main_row['mass_divergence'],
            fraction_of_balanced_risk_removable_by_exact_query_calibration=(
                main_row['mass_divergence'] / main_row['query_balanced_divergence']),
            raw_L2_reduction_vs_favor=(1 - main_row['relative_squared_risk'] /
                                       rows['favor_plus']['relative_squared_risk']),
            raw_I_reduction_vs_favor=(1 - main_row['relative_I_divergence'] /
                                      rows['favor_plus']['relative_I_divergence']),
            scope='Four-head means on saved full empirical products, fixed seed11; '
                  'oracle fraction is balanced I risk, not raw unweighted I or L2. '
                  'No oracle-calibrated deployment model was trained.'
        ))

    data = torch.load(P / 'results/data_fresh_wiki3.pt', weights_only=True, mmap=True)
    # Separate documents on the two sides. Selection is independent of scores.
    q = data['q'][:8, :, 512::32].permute(1, 0, 2, 3).flatten(1, 2).double()
    k = data['k'][8:16, :, :512:32].permute(1, 0, 2, 3).flatten(1, 2).double()
    fit_path = P / 'fits/factorized_both_m64_s11_n4096.pt'
    box = torch.load(fit_path, weights_only=True)
    sd = box['state_dict']
    norm = {key: sd[key] for key in ['q_mean', 'q_std', 'k_mean', 'k_std']}
    net = Pair(norm, m=64, variant='factorized_both').double()
    net.load_state_dict(sd)
    net.eval()
    scale = torch.tensor(box['metadata']['log_scale'], dtype=torch.float64)
    with torch.inference_mode():
        fq = (net.log_feature(q, 'q') + scale[:, None, None] / 2).exp()
        fk = (net.log_feature(k, 'k') + scale[:, None, None] / 2).exp()
        g = fq @ fk.transpose(-1, -2)
        t = (q @ k.transpose(-1, -2) / math.sqrt(128)).exp()
        z = t.mean(-1)
        total = g.mean(-1)
        mu = fk.mean(1)
        pi = fq * mu[:, None] / total[:, :, None]
        psi = fk / mu[:, None]
        canonical = total[:, :, None] * (pi @ psi.transpose(-1, -2))

        a = torch.linspace(-.7, .7, q.shape[1], dtype=torch.float64).exp()[None]
        oracle = g * (z / total)[:, :, None]
        lhs = divergence(t, g * a[:, :, None]).mean(-1)
        residual = divergence(t, oracle).mean(-1)
        amplitude = divergence(z, a * total)

        # Restore the original L2 spectrum through the Z^2-weighted Q measure.
        p = t / z[:, :, None]
        m2 = z.square().mean(-1)
        wq2 = z.square() / (m2[:, None] * q.shape[1])
        weighted_direction = p * wq2.sqrt()[:, :, None] / math.sqrt(k.shape[1])
        raw_sv = torch.linalg.svdvals(t / math.sqrt(q.shape[1] * k.shape[1]))
        direction_sv = torch.linalg.svdvals(weighted_direction) * m2.sqrt()[:, None]

        # A fixed subset, arbitrary V: calibration preserves output even when
        # the actual context differs from the reference key bank.
        gen = torch.Generator().manual_seed(731)
        v = torch.randn(g.shape[0], 31, 7, dtype=torch.float64, generator=gen)
        context_g = g[:, :, 7:38]
        context_h = oracle[:, :, 7:38]
        out_g = (context_g / context_g.sum(-1, keepdim=True)) @ v
        out_h = (context_h / context_h.sum(-1, keepdim=True)) @ v
        checks = dict(
            canonical_representation_relative_error=relative_max(canonical, g),
            simplex_sum_absolute_error=float((pi.sum(-1) - 1).abs().max()),
            component_mean_absolute_error=float((psi.mean(1) - 1).abs().max()),
            raw_I_projection_relative_error=relative_max(lhs, residual + amplitude),
            weighted_L2_spectrum_relative_error=relative_max(raw_sv, direction_sv),
            arbitrary_context_output_relative_error=relative_max(out_h, out_g),
        )
        assert all(value < 1e-10 for value in checks.values()), checks

    sources = [summary_path, fit_path, Path(__file__).resolve()]
    result = dict(
        scope='Existing-result reanalysis plus FP64 algebra checks, not new '
              'training, benchmark, population inference, or larger-scale validation.',
        full_product_reanalysis=reviewed,
        identity_check_protocol=dict(heads=4, q_vectors_per_head=q.shape[1],
                                     k_vectors_per_head=k.shape[1],
                                     q_documents=list(range(8)),
                                     k_documents=list(range(8, 16)),
                                     data='data_fresh_wiki3.pt', dtype='float64', device='cpu'),
        identity_checks=checks,
        sha256={str(path.relative_to(P)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sources},
    )
    output = P / 'checks/amplitude_direction_focus.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
