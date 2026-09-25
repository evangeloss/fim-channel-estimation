"""Copy to tests/; run with --run-dir pointing to the completed training folder.

Example from tests/:
python evaluate_deformation_sweep.py --run-dir ../artifacts/training_20260925_135010_277118

No training. Saves sample NMSE, paired environment means, summary CSV/JSON and
an optional plot. Requires the existing project, NumPy and PyTorch; Matplotlib
is optional. Lower NMSE is better.
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import dataset.generator as generator
from model.estimator import ChannelEstimator
from training.trainer import nmse_per_sample, nmse_db


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--channels', type=int, default=50, help='New physical environments, shared across amplitudes.')
    parser.add_argument('--seed', type=int, default=98765)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--b-values', type=float, nargs='+', default=[0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5])
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    if args.channels < 1 or args.batch_size < 1 or args.seed < 0:
        parser.error('Require positive channels/batch-size and a nonnegative seed.')
    if any(not np.isfinite(q) or q < 0 for q in args.b_values):
        parser.error('Deformation amplitudes must be finite and nonnegative.')
    q_values = sorted(set(args.b_values))
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / 'experiment_config.json').read_text(encoding='utf-8'))
    used_seeds = [s['seed'] for s in config['splits'].values()]
    if args.seed in used_seeds:
        parser.error('Use a new seed, distinct from the original dataset split seeds.')
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(min(4, torch.get_num_threads()))
    checkpoint = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=True)
    model_config = dict(checkpoint['model_config'])
    model_config.setdefault('residual_reference', 'first')  # Legacy checkpoints
    model = ChannelEstimator(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    simulation = config['simulation'].copy()
    simulation['residual_reference'] = model_config['residual_reference']
    m_count = simulation['n_deformations']
    k_pairs = simulation['n_subcarriers'] - 1
    output = run_dir / datetime.now().strftime('deformation_sweep_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    methods = ('model', 'first_observation', 'mean_observation')
    summary, environment_rows, sample_rows = [], [], []
    path_reference, pattern_reference = [], []
    original_paths = generator.generate_path_parameters
    original_codebook = generator.generate_deformation_codebook

    # The same seed and unchanged RNG call sequence match paths, pilot draws,
    # deformation patterns and standardized noise. Audit paths and patterns
    # explicitly so future generator changes cannot silently break that match.
    for sweep_index, q in enumerate(q_values):
        current_paths, current_patterns = [], []

        def capture_paths(*positional, **keywords):
            paths = original_paths(*positional, **keywords)
            current_paths.append({key: np.array(value, copy=True) for key, value in paths.items()})
            return paths

        def capture_codebook(*positional, **keywords):
            result = original_codebook(*positional, **keywords)
            if q > 0:
                current_patterns.append((np.stack(result[1])/q, np.stack(result[3])/q))
            return result

        print(f'Generating matched sweep b/lambda={q:.3f}...', flush=True)
        with patch.object(generator, 'generate_path_parameters', side_effect=capture_paths), \
             patch.object(generator, 'generate_deformation_codebook', side_effect=capture_codebook):
            arrays = generator.generate_dataset(**simulation, n_channels=args.channels,
                                                 b_min=q, b_max=q, seed=args.seed)
        assert len(current_paths) == args.channels
        if sweep_index == 0:
            path_reference = current_paths
        else:
            for actual, expected in zip(current_paths, path_reference):
                for key in expected:
                    np.testing.assert_array_equal(actual[key], expected[key], err_msg='Paths changed across sweep.')
        if q > 0:
            assert len(current_patterns) == args.channels
            if not pattern_reference:
                pattern_reference = current_patterns
            else:
                for actual_pair, expected_pair in zip(current_patterns, pattern_reference):
                    for actual, expected in zip(actual_pair, expected_pair):
                        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12,
                                                   err_msg='Deformation patterns changed across sweep.')
        observations, geometry, residual_targets = arrays
        expected_samples = args.channels*k_pairs
        assert observations.shape[0] == expected_samples
        dataset = TensorDataset(*(torch.as_tensor(a, dtype=torch.float32) for a in arrays))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        accumulated = {name: [] for name in methods}
        with torch.no_grad():
            for obs, geo, residual in loader:
                obs, geo, residual = [a.to(device) for a in (obs, geo, residual)]
                true_h0 = model.reconstruction.reference(obs) + residual
                # Average COMPLEX estimates by averaging corresponding real and
                # imaginary channels, not magnitudes. All M share one sample scale.
                mean = obs.reshape(obs.shape[0], m_count, 4, obs.shape[2], obs.shape[3]).mean(dim=1)
                predictions = dict(model=model(obs, geo), first_observation=obs[:, :4], mean_observation=mean)
                for name, prediction in predictions.items():
                    values = nmse_per_sample(prediction, true_h0)
                    if not torch.isfinite(values).all().item():
                        raise FloatingPointError(f'Nonfinite {name} NMSE at b={q}.')
                    accumulated[name].append(values.cpu().numpy())
        accumulated = {name: np.concatenate(parts) for name, parts in accumulated.items()}
        row = dict(b_over_lambda=q, physical_environments=args.channels, samples=expected_samples)
        for name, values in accumulated.items():
            row[name+'_nmse'] = float(values.mean())
            row[name+'_db'] = nmse_db(row[name+'_nmse'])
        row['gain_over_first_db'] = row['first_observation_db'] - row['model_db']
        row['gain_over_mean_db'] = row['mean_observation_db'] - row['model_db']
        summary.append(row)
        for index in range(expected_samples):
            sample_rows.append(dict(b_over_lambda=q, environment=index//k_pairs,
                subcarrier_pair=index % k_pairs,
                **{name+'_nmse': float(values[index]) for name, values in accumulated.items()}))
        for environment in range(args.channels):
            selection = slice(environment*k_pairs, (environment+1)*k_pairs)
            environment_rows.append(dict(b_over_lambda=q, environment=environment,
                **{name+'_nmse': float(values[selection].mean()) for name, values in accumulated.items()}))
        print(f'Model {row["model_db"]:.2f} dB | first {row["first_observation_db"]:.2f} dB | '
              f'mean {row["mean_observation_db"]:.2f} dB | gain over mean {row["gain_over_mean_db"]:+.2f} dB', flush=True)
        # Save progress so completed amplitudes remain available if interrupted.
        write_csv(output / 'summary.csv', summary)
        write_csv(output / 'environment_nmse.csv', environment_rows)
        write_csv(output / 'sample_nmse.csv', sample_rows)
        del loader, dataset, arrays, observations, geometry, residual_targets

    metadata = dict(checkpoint=str(run_dir/'best.pt'), checkpoint_epoch=checkpoint['epoch'],
        seed=args.seed, channels=args.channels, simulation=simulation, b_values=q_values,
        device=str(device), training_range=[config['splits']['train']['b_min'], config['splits']['train']['b_max']],
        matching='Identical seed and RNG call sequence. Path equality and scaled deformation patterns checked.',
        noise='Common standardized random draws; noise variance follows signal power at each geometry.',
        metric='dB of mean linear per-sample NMSE over adjacent-subcarrier pairs; lower is better.',
        interpretation='Evaluation only. Averaging is a complex arithmetic mean, with no geometry compensation. '
                       'No claim that modulation contributes independently; that requires a retrained ablation.')
    (output/'sweep_config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Matplotlib unavailable; CSV/JSON results saved. Install matplotlib and rerun for a plot.')
    else:
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        labels = dict(model='Trained model', first_observation='First observation', mean_observation='Mean of all observations')
        for name in methods:
            ax.plot(q_values, [r[name+'_db'] for r in summary], marker='o', label=labels[name])
        lo, hi = metadata['training_range']
        ax.axvspan(lo, hi, color='gray', alpha=.12, label='Training deformation range')
        ax.set(xlabel='b / wavelength', ylabel='NMSE (dB), lower is better',
               title=f'Matched deformation sweep | {args.channels} new environments')
        ax.grid(alpha=.25)
        ax.legend()
        fig.savefig(output/'nmse_vs_deformation.png', dpi=170)
        plt.close(fig)
    print('Saved sweep results:', output.resolve(), flush=True)


if __name__ == '__main__':
    main()
