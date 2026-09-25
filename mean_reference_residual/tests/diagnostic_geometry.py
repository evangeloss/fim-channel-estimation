"""Place in tests/; run python diagnose_geometry.py (NumPy + Matplotlib).

Uses existing channel/ and geometry/ code without editing it.
Optional: --project-root PATH --output PATH --realizations 20 --seed 42
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def correlation(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= 1e-30:
        return float('nan')
    return float(np.clip(abs(np.vdot(a, b)) / denominator, 0, 1))


def relative_error(a, reference):
    return float(np.linalg.norm(a - reference) / max(np.linalg.norm(reference), 1e-30))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--realizations', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.realizations < 1:
        parser.error('--realizations must be positive')
    sys.path.insert(0, str(args.project_root.resolve()))
    from channel import (build_channel, steering_vector_fim, generate_path_parameters,
                         generate_multi_pilots, transmit_pilots, ridge_estimate)
    from geometry.surfaces import generate_reference_geometry, generate_deformation_codebook

    output = args.output or args.project_root / 'diagnostic_results'
    output.mkdir(parents=True, exist_ok=True)
    wl, fs, k_count, m_count = 3e8 / 28e9, 1e5, 4, 8
    q_values = np.linspace(0, 0.5, 11)  # q = b / wavelength
    snr_db, step = 10.0, 1e-4
    pb, pu, nb, nu = generate_reference_geometry(5, 5, wl/8, wl/8, 5, 5, wl/8, wl/8, wl)
    zeros_b, zeros_u = np.zeros_like(pb), np.zeros_like(pu)
    rng = np.random.default_rng(args.seed)
    noise_rng = np.random.default_rng(args.seed + 1)
    rows, matrices = [], []
    checks = {}

    # Independent analytical phase-ratio check for a single element.
    az, el = 0.37, 0.21
    direction = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    displacement = zeros_b.copy()
    displacement[:, 0] = 0.03 * wl * nb
    a0 = steering_vector_fim(pb, zeros_b, wl, az, el)
    a1 = steering_vector_fim(pb, displacement, wl, az, el)
    np.testing.assert_allclose(a1 / a0, np.exp(2j*np.pi/wl*(direction @ displacement)), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(a1), 1.0, atol=1e-12)
    checks['steering_phase_and_unit_norm'] = 'passed'

    # Correlation deliberately ignores a common phase, unlike relative error.
    example = np.array([1+2j, 3-1j])
    assert np.isclose(correlation(example, 1j*example), 1)
    assert relative_error(1j*example, example) > 1

    max_step_disagreement = 0.0
    for realization in range(args.realizations):
        paths = generate_path_parameters(3, fs, rng)
        # Draw patterns once at q=1, then scale them. Never redraw paths or
        # deformation patterns within an amplitude sweep.
        _, patterns_b, _, patterns_u = generate_deformation_codebook(pb, pu, nb, nu, wl, 1.0, m_count, rng)
        pilots = generate_multi_pilots(pu.shape[1], 1, rng)

        def channel(q, m):
            return build_channel(paths, pb, q*patterns_b[m], pu, q*patterns_u[m], wl, fs, k_count)

        h0 = build_channel(paths, pb, zeros_b, pu, zeros_u, wl, fs, k_count)
        np.testing.assert_allclose(channel(0, 0), h0, atol=1e-12)
        for q in q_values:
            clean = [channel(q, m) for m in range(m_count)]
            estimates = []
            for h in clean:
                y, variance = transmit_pilots(h, pilots, snr_db, noise_rng)
                estimates.append(np.stack([ridge_estimate(y[:, :, k, :], pilots, variance)
                                           for k in range(k_count)], axis=-1))
            # Same sample-wide positive scaling as the dataset, for this
            # four-subcarrier diagnostic. Correlation is scale invariant.
            clean_corr = [correlation(h, h0) for h in clean]
            errors = [relative_error(h, h0) for h in clean]
            sensitivities = []
            for m in range(m_count):
                # Directional derivative dH/dq at fixed deformation pattern.
                # Negative q at zero is a valid signed normal displacement.
                d1 = (channel(q + step, m) - channel(q - step, m)) / (2*step)
                d2 = (channel(q + step/2, m) - channel(q - step/2, m)) / step
                disagreement = relative_error(d1, d2)
                max_step_disagreement = max(max_step_disagreement, disagreement)
                sensitivities.append(float(np.linalg.norm(d2) / max(np.linalg.norm(h0), 1e-30)))
            pair_clean = [correlation(clean[i], clean[j]) for i in range(m_count) for j in range(i+1, m_count)]
            pair_noisy = [correlation(estimates[i], estimates[j]) for i in range(m_count) for j in range(i+1, m_count)]
            rows.append(dict(realization=realization, b_over_lambda=float(q),
                corr_reference=float(np.mean(clean_corr)),
                relative_change=float(np.mean(errors)),
                sensitivity_per_b_over_lambda=float(np.mean(sensitivities)),
                corr_between_clean=float(np.mean(pair_clean)),
                corr_between_estimates=float(np.mean(pair_noisy)),
                estimate_error_to_own_channel=float(np.mean([relative_error(e,h) for e,h in zip(estimates,clean)]))))
            if realization == 0 and (np.isclose(q, 0.1) or np.isclose(q, 0.5)):
                channels = [h0] + clean
                matrix = np.array([[correlation(a,b) for b in channels] for a in channels])
                np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
                np.testing.assert_allclose(np.diag(matrix), 1, atol=1e-12)
                matrices.append((q, matrix))
        print(f'Completed environment {realization+1}/{args.realizations}', flush=True)

    assert max_step_disagreement < 1e-4, 'Finite-difference sensitivity did not converge.'
    checks['zero_deformation_identity'] = 'passed'
    checks['correlation_symmetry_diagonal_and_phase_invariance'] = 'passed'
    checks['max_relative_derivative_step_disagreement'] = max_step_disagreement
    write_csv(output / 'measurements.csv', rows)
    keys = ['corr_reference', 'relative_change', 'sensitivity_per_b_over_lambda',
            'corr_between_clean', 'corr_between_estimates', 'estimate_error_to_own_channel']
    summary = []
    for q in q_values:
        selected = [r for r in rows if np.isclose(r['b_over_lambda'], q)]
        entry = {'b_over_lambda': float(q)}
        for key in keys:
            values = np.array([r[key] for r in selected])
            entry[key+'_mean'] = float(values.mean())
            entry[key+'_p10'] = float(np.percentile(values, 10))
            entry[key+'_p90'] = float(np.percentile(values, 90))
        summary.append(entry)
    write_csv(output / 'summary.csv', summary)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, key, title in zip(axes.flat[:3], keys[:3],
        ['Correlation with undeformed channel', 'Relative channel change (not squared)', 'Local sensitivity per unit b / wavelength']):
        ax.plot(q_values, [r[key+'_mean'] for r in summary], marker='o')
        ax.fill_between(q_values, [r[key+'_p10'] for r in summary], [r[key+'_p90'] for r in summary], alpha=.2)
        ax.set_title(title)
    ax = axes.flat[3]
    for key, label in [('corr_between_clean','Clean channels'), ('corr_between_estimates','Tentative estimates, 10 dB')]:
        ax.plot(q_values, [r[key+'_mean'] for r in summary], marker='o', label=label)
    ax.set_title('Correlation between different deformation states')
    ax.legend()
    for ax in axes.flat:
        ax.set_xlabel('b / wavelength')
        ax.grid(alpha=.25)
    axes.flat[0].set_ylim(0, 1.02)
    axes.flat[3].set_ylim(0, 1.02)
    fig.suptitle('Fixed paths and fixed deformation patterns within each sweep\nMeans; shaded bands: 10–90% of environment means')
    fig.savefig(output / 'geometry_sweep.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(matrices), figsize=(11, 5), constrained_layout=True, squeeze=False)
    labels = ['H0'] + [f'H{m+1}' for m in range(m_count)]
    for ax, (q, matrix) in zip(axes.flat, matrices):
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap='viridis')
        ax.set_xticks(range(m_count+1), labels, rotation=45)
        ax.set_yticks(range(m_count+1), labels)
        ax.set_title(f'Example environment: b / wavelength = {q:.1f}')
        np.savetxt(output / f'correlation_b_{q:.1f}.csv', matrix, delimiter=',', header=','.join(labels), comments='')
    fig.colorbar(im, ax=list(axes.flat), label='Absolute normalized complex inner product')
    fig.savefig(output / 'correlation_matrices.png', dpi=160)
    plt.close(fig)
    config = dict(seed=args.seed, environments=args.realizations, fc=28e9, fs=fs,
        subcarriers=k_count, deformations=m_count, paths=3, bs_shape=[5,5], ue_shape=[5,5],
        spacing_over_wavelength=0.125, snr_db=snr_db, finite_difference_step=step,
        b_over_lambda=q_values.tolist(), checks=checks)
    (output / 'config_and_checks.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print('Diagnostics and analytical consistency checks completed:', output.resolve())
    print('These assess the implemented phase-only path model, not real-world physical validity.')


if __name__ == '__main__':
    main()
