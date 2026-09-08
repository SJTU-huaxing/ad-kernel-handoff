"""Aggregate completed scans; generate portable figures and numerical audits.

Run with a Python environment containing NumPy and Matplotlib. No GPU scan,
model fitting, or held-out parameter selection is performed here.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

P = Path(__file__).resolve().parent
DATASETS = ['moment_train', 'internal', 'official', 'disjoint_ab', 'disjoint_ba']
HEADS = [(14, 0), (14, 6), (27, 0), (27, 6)]
RANKS = [16, 32, 64, 128]


def read(name):
    return json.loads((P / name).read_text())


def write(name, obj):
    (P / name).write_text(json.dumps(obj, indent=2, allow_nan=False))


def head_name(h):
    return f'L{h[0]}H{h[1]}'


summaries = {ds: read(f'results/{ds}_summary.json') for ds in DATASETS}
galerkin = read('results/train_galerkin_results.json')
gindex = {(r['dataset'], r['n'], tuple(r['head']), r['m']): r['nmse']
          for r in galerkin['results']}
rows = []
max_identity = 0.0
bound_checks = risk_checks = monotonic_checks = 0
for ds, blocks in summaries.items():
    for block in blocks:
        n = block['samples_per_marginal']
        assert block['pairs_per_head'] == n*n
        for h in block['heads']:
            previous_projection = float('inf')
            for m in RANKS:
                bound = h['signed_floor_bracket'][str(m)]
                low, upper = bound['lower'], bound['upper']
                assert -1e-8 <= low <= upper + 1e-8 and upper <= 1 + 1e-8
                assert abs(upper-low-h['compression_residual_fraction']) < 1e-8
                bound_checks += 1
                proj = h['partition_projection'][str(m)]['nmse']
                assert -1e-8 <= proj <= previous_projection + 1e-8
                previous_projection = proj
                monotonic_checks += 1
                positive = h['positive'].get(str(m), {})
                signed = gindex[(ds, n, tuple(h['head']), m)]
                assert signed + 1e-8 >= low
                risk_checks += 1
                if positive:
                    error = abs(positive['nmse'] - proj - positive['conditional_mean_estimation_gap'])
                    max_identity = max(max_identity, error)
                    assert error < 1e-8 and positive['nmse'] + 1e-8 >= low
                    assert positive['conditional_mean_estimation_gap'] >= -1e-8
                    risk_checks += 1
                rows.append(dict(
                    dataset=ds, samples_per_marginal=n, pairs_per_head=n*n,
                    head=head_name(h['head']), m=m, signed_floor_lower=low,
                    signed_floor_upper=upper, compression_residual_fraction=upper-low,
                    positive_nmse=positive.get('nmse'), partition_projection_nmse=proj,
                    mean_estimation_gap=positive.get('conditional_mean_estimation_gap'),
                    positive_idiv_per_mass=positive.get('idiv_per_mass'),
                    train_galerkin_signed_nmse=signed,
                    train_anchor_nystrom_signed_nmse=h['train_nystrom'][str(m)]['nmse'],
                    log_raw_second_moment=h['log_raw_kernel_second_moment']))

with (P / 'curves.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

protocol = read('results/protocol.json')
assert not set(protocol['calibration_doc_ordinals']) & set(protocol['moment_fit_doc_ordinals'])
assert len(set(protocol['calibration_doc_ordinals']) | set(protocol['moment_fit_doc_ordinals'])) == 4096
coefficients = read('results/positive_kernel_coefficients.json')
assert coefficients['no_test_fitting']
coefficient_min = min(np.asarray(h[str(m)]).min() for h in coefficients['coefficients'] for m in RANKS)
assert coefficient_min > 0
empty_training_cells = sum(
    np.count_nonzero(np.asarray(h['partition_projection'][str(m)]['cell_mass']) == 0)
    for h in summaries['moment_train'][-1]['heads'] for m in RANKS)
audit = dict(
    curve_rows=len(rows), spectral_bracket_checks=bound_checks,
    rank_risk_lower_bound_checks=risk_checks, nested_partition_projection_checks=monotonic_checks,
    max_relative_pythagorean_error=max_identity,
    max_train_galerkin_projection_error=galerkin['max_training_projection_identity_error'],
    minimum_scaled_positive_coefficient=float(coefficient_min),
    empty_moment_fit_cells=int(empty_training_cells),
    train_calibration_and_moment_documents_disjoint=True,
    internal_pairs_per_head=protocol['exhaustive_internal_pairs_per_head'],
    internal_pairs_all_heads=protocol['exhaustive_internal_pairs_all_heads'],
    status='passed',
    scope='Numerical empirical-operator and frozen-feature checks; not a population confidence certificate.')
write('checks/aggregate_audit.json', audit)

full = {ds: [r for r in rows if r['dataset'] == ds and
             r['samples_per_marginal'] == summaries[ds][-1]['samples_per_marginal']]
        for ds in DATASETS}
write('summary.json', dict(
    main_m64={ds: [r for r in rr if r['m'] == 64] for ds, rr in full.items()},
    audit=audit,
    interpretation={
        'general_schmidt_framework': 'Valid under the Hilbert-Schmidt assumption; identities verified numerically.',
        'covariance_only_prediction': 'Not valid without distributional assumptions; bounded exact counterexample provided.',
        'gaussian_real_llm_prediction': 'Not established; previous 16-head experiment had only 4 HS-valid Gaussian surrogates.',
        'new_kernel': 'Nonnegative conditional-mean projection on training spectral partitions; no FAVOR Gaussian feature identity.',
        'population_identification': 'Not established. Exhaustive empirical products and disjoint-document folds do not identify unknown population tails.',
        'near_optimal_positive_features': 'Not established. Spectral brackets are wide; partition approximation dominates current risk.',
        'kan_specific_advantage': 'Not established. No new KAN/MLP comparison in this operator experiment.'}))

figdir = P / 'figures'
figdir.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
for ax, h in zip(axes.flat, HEADS):
    rr = [r for r in full['internal'] if r['head'] == head_name(h)]
    lo = np.array([r['signed_floor_lower'] for r in rr])
    up = np.array([r['signed_floor_upper'] for r in rr])
    ax.fill_between(RANKS, lo, up, color='#aeb8cb', alpha=.38, label='Empirical signed optimum bracket')
    ax.plot(RANKS, up, color='#687d9c', lw=1)
    ax.plot(RANKS, lo, color='#687d9c', lw=1)
    ax.plot(RANKS, [r['positive_nmse'] for r in rr], 'o-', color='#cf6327', label='Frozen nonnegative kernel')
    ax.plot(RANKS, [r['partition_projection_nmse'] for r in rr], '--', color='#cf6327', label='Best means in fixed partitions (diagnostic)')
    ax.plot(RANKS, [r['train_galerkin_signed_nmse'] for r in rr], 's-', color='#18786a', label='Frozen train-Galerkin kernel (signed)')
    ax.set_title(head_name(h))
    ax.set_xscale('log', base=2)
    ax.set_xticks(RANKS, labels=RANKS)
    ax.set_ylim(0, 1)
    ax.grid(alpha=.15)
for ax in axes[-1]: ax.set_xlabel('Feature dimension m')
for ax in axes[:, 0]: ax.set_ylabel('Relative raw-kernel squared error')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9, bbox_to_anchor=(.5, -.04))
fig.suptitle('Full held-out empirical product: 131,072 Q x 131,072 K per head\nShaded bounds concern the empirical operator, not unknown population confidence intervals', fontsize=12)
fig.tight_layout(rect=(0, .06, 1, .91))
fig.savefig(figdir / 'rank_curves.png')
fig.savefig(figdir / 'rank_curves.pdf')
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for ax, ds, title in zip(axes, ['disjoint_ab', 'disjoint_ba'], ['Q documents A x K documents B', 'Q documents B x K documents A']):
    rr = [r for r in full[ds] if r['m'] == 64]
    x = np.arange(4)
    low = np.array([r['signed_floor_lower'] for r in rr]); up = np.array([r['signed_floor_upper'] for r in rr])
    ax.vlines(x-.2, low, up, color='#687d9c', linewidth=7, alpha=.5, label='Empirical signed optimum bracket')
    ax.scatter(x, [r['positive_nmse'] for r in rr], color='#cf6327', marker='o', label='Frozen nonnegative kernel')
    ax.scatter(x+.2, [r['train_galerkin_signed_nmse'] for r in rr], color='#18786a', marker='s', label='Frozen signed kernel')
    ax.set_xticks(x, [r['head'] for r in rr]); ax.set_title(title)
    ax.grid(axis='y', alpha=.15); ax.set_ylim(0, 1)
axes[0].set_ylabel('Relative raw-kernel squared error')
fig.suptitle('Disjoint-document products, m=64; 65,536 vectors on each side\nTraining and all deployed parameters are unchanged', fontsize=12)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9)
fig.tight_layout(rect=(0, .1, 1, .88))
fig.savefig(figdir / 'disjoint_documents.png')
fig.savefig(figdir / 'disjoint_documents.pdf')
plt.close(fig)

diag = read('results/distribution_diagnostics.json')
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for h in HEADS:
    rr = [r for r in rows if r['dataset'] == 'internal' and r['head'] == head_name(h) and r['m'] == 64]
    x = [r['samples_per_marginal'] for r in rr]
    logh = np.array([r['log_raw_second_moment'] for r in rr])
    axes[0].plot(x, np.exp(logh-logh[-1]), 'o-', label=head_name(h))
axes[0].set_xscale('log', base=2); axes[0].set_yscale('log')
axes[0].set_xticks([8192, 32768, 131072], labels=['8,192', '32,768', '131,072'])
axes[0].axhline(1, color='gray', linestyle=':', lw=1)
axes[0].set_xlabel('Vectors per marginal (same 256 documents)')
axes[0].set_ylabel('Kernel second moment / full-sample value')
axes[0].set_title('Adding token positions changes tail coverage')
axes[0].legend(fontsize=9)
dd = [r for r in diag['document_sensitivity'] if r['n'] == 131072]
perc = np.array([r['bootstrap_second_moment_ratio_percentiles'] for r in dd])
x = np.arange(4)
axes[1].errorbar(x, perc[:, 1], yerr=np.stack([perc[:, 1]-perc[:, 0], perc[:, 2]-perc[:, 1]]), fmt='o', capsize=5, color='#687d9c')
axes[1].axhline(1, color='gray', linestyle=':', lw=1)
axes[1].set_xticks(x, [head_name(r['head']) for r in dd])
axes[1].set_ylabel('Resampled second moment / original value')
axes[1].set_title('Document bootstrap: 2.5 / 50 / 97.5 percentiles')
for ax in axes: ax.grid(alpha=.15)
fig.suptitle('Sensitivity diagnostics; bootstrap ranges are not certified population intervals', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, .9))
fig.savefig(figdir / 'distribution_sensitivity.png')
fig.savefig(figdir / 'distribution_sensitivity.pdf')
plt.close(fig)
print(json.dumps(audit, indent=2))
