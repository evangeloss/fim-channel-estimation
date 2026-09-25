"""Copy into tests/; run with --run-dir ../artifacts/YOUR_TRAINING_RUN.

No training. Uses fresh matched samples to diagnose a saved best checkpoint.
Works with first-reference and updated mean-reference project versions.
"""
import argparse
import csv
import inspect
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from dataset.generator import generate_dataset
from model.estimator import ChannelEstimator


def stats(values):
    x = np.asarray(values, dtype=float).reshape(-1)
    return dict(mean=float(x.mean()), std=float(x.std()), minimum=float(x.min()),
                p05=float(np.percentile(x, 5)), median=float(np.median(x)),
                p95=float(np.percentile(x, 95)), maximum=float(x.max()))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--channels', type=int, default=20)
    parser.add_argument('--b', type=float, default=None, help='Fixed b/wavelength; otherwise original validation range.')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=87654)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    if args.channels < 2 or args.batch_size < 1 or args.seed < 0:
        parser.error('Require at least two environments, positive batch size and nonnegative seed.')
    if args.b is not None and (not np.isfinite(args.b) or args.b < 0):
        parser.error('--b must be finite and nonnegative.')
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir/'experiment_config.json').read_text(encoding='utf-8'))
    if args.seed in [s['seed'] for s in config['splits'].values()]:
        parser.error('Choose a seed different from the original training/validation/test seeds.')
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(min(4, torch.get_num_threads()))
    checkpoint = torch.load(run_dir/'best.pt', map_location='cpu', weights_only=True)
    model_config = dict(checkpoint['model_config'])
    reference_kind = model_config.get('residual_reference', 'first')
    if 'residual_reference' in inspect.signature(ChannelEstimator).parameters:
        model_config['residual_reference'] = reference_kind
    elif reference_kind != 'first':
        raise ValueError('Mean-reference checkpoint requires the updated estimator.')
    model = ChannelEstimator(**model_config).to(device).eval()
    model.load_state_dict(checkpoint['model_state_dict'])
    simulation = config['simulation'].copy()
    if 'residual_reference' in inspect.signature(generate_dataset).parameters:
        simulation['residual_reference'] = reference_kind
    else:
        simulation.pop('residual_reference', None)
        if reference_kind != 'first':
            raise ValueError('Update the dataset generator for mean-reference targets.')
    low, high = (config['splits']['validation'][k] for k in ('b_min', 'b_max'))
    if args.b is not None:
        low = high = args.b
    print(f'Checkpoint epoch {checkpoint["epoch"]}; reference={reference_kind}; b/lambda={low}..{high}', flush=True)
    print(f'Generating {args.channels} new environments...', flush=True)
    x, g, t = generate_dataset(**simulation, n_channels=args.channels,
                               b_min=low, b_max=high, seed=args.seed)
    pairs = simulation['n_subcarriers']-1
    assert len(x) == args.channels*pairs
    m_count = simulation['n_deformations']
    # Derangement at ENVIRONMENT level, not adjacent-pair sample level.
    # A cyclic shift of a random ordering has no self matches.
    rng = np.random.default_rng(args.seed+1)
    order = rng.permutation(args.channels)
    donors = np.empty(args.channels, dtype=int)
    donors[order] = np.roll(order, 1)
    donor_samples = donors[np.arange(len(x))//pairs]*pairs + np.arange(len(x))%pairs
    np.testing.assert_array_equal(donor_samples//pairs != np.arange(len(x))//pairs, True)
    output = run_dir/datetime.now().strftime('modulation_diagnostic_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    metrics = {key: [] for key in ('gamma', 'beta', 'scale_ratio', 'bias_ratio', 'change_ratio', 'feature_rms')}
    errors = {key: [] for key in ('alpha_0', 'alpha_0.5', 'alpha_1', 'shuffled_geometry', 'first_observation', 'mean_observation')}
    alphas = (0.0, 0.5, 1.0)

    def nmse(pred, truth):
        return ((pred-truth).square().sum((1,2,3)) / truth.square().sum((1,2,3)).clamp_min(1e-12))

    with torch.no_grad():
        for start in range(0, len(x), args.batch_size):
            end = min(start+args.batch_size, len(x))
            obs, geo, target = [torch.as_tensor(a[start:end], device=device) for a in (x,g,t)]
            wrong_geo = torch.as_tensor(g[donor_samples[start:end]], device=device)
            mean = obs.reshape(len(obs), m_count, 4, obs.shape[2], obs.shape[3]).mean(1)
            reference = mean if reference_kind == 'mean' else obs[:, :4]
            truth = reference+target
            features = model.cnn(model.beamspace(obs))
            conditioned, gamma, beta = model.modulation(features, model.geometry_encoder.encode_grid(geo), return_parameters=True)
            norm = features.flatten(1).norm(dim=1).clamp_min(1e-12)
            scale = (gamma-1)*features
            bias = beta.expand_as(features)
            for key, value in dict(gamma=gamma.flatten(1), beta=beta.flatten(1),
                scale_ratio=scale.flatten(1).norm(dim=1)/norm,
                bias_ratio=bias.flatten(1).norm(dim=1)/norm,
                change_ratio=(conditioned-features).flatten(1).norm(dim=1)/norm,
                feature_rms=features.square().flatten(1).mean(1).sqrt()).items():
                metrics[key].append(value.cpu().numpy())
            for alpha, name in zip(alphas, ('alpha_0','alpha_0.5','alpha_1')):
                adjusted = features + alpha*(conditioned-features)
                prediction = model.reconstruction(model.decoder(adjusted), obs)
                errors[name].append(nmse(prediction, truth).cpu().numpy())
                if alpha == 1 and start == 0:
                    torch.testing.assert_close(prediction, model(obs, geo), rtol=2e-4, atol=2e-5)
            shuffled = model.modulation(features, model.geometry_encoder.encode_grid(wrong_geo))
            prediction = model.reconstruction(model.decoder(shuffled), obs)
            errors['shuffled_geometry'].append(nmse(prediction, truth).cpu().numpy())
            errors['first_observation'].append(nmse(obs[:, :4], truth).cpu().numpy())
            errors['mean_observation'].append(nmse(mean, truth).cpu().numpy())
            print(f'Evaluated {end}/{len(x)} samples', flush=True)
    metrics = {key: np.concatenate(value) for key,value in metrics.items()}
    errors = {key: np.concatenate(value) for key,value in errors.items()}
    if not all(np.isfinite(a).all() for a in [*metrics.values(), *errors.values()]):
        raise FloatingPointError('Nonfinite diagnostic values.')
    # Bootstrap environment means, not correlated adjacent-subcarrier pairs.
    env_errors = {key: value.reshape(args.channels,pairs).mean(1) for key,value in errors.items()}
    boot = np.random.default_rng(args.seed+2).integers(0,args.channels,size=(2000,args.channels))
    normal = env_errors['alpha_1']
    results = []
    for name, values in env_errors.items():
        delta = 10*np.log10(np.maximum(values[boot].mean(1),1e-30)/np.maximum(normal[boot].mean(1),1e-30))
        results.append(dict(condition=name, nmse=float(values.mean()), nmse_db=float(10*np.log10(max(values.mean(),1e-30))),
            delta_vs_normal_db=float(10*np.log10(max(values.mean(),1e-30)/max(normal.mean(),1e-30))),
            paired_bootstrap_delta_low=float(np.percentile(delta,2.5)),
            paired_bootstrap_delta_high=float(np.percentile(delta,97.5))))
    write_csv(output/'performance.csv', results)
    write_csv(output/'sample_metrics.csv', [dict(sample=i, environment=i//pairs,
        **{key: float(value[i]) for key,value in metrics.items() if value.ndim==1},
        **{key+'_nmse': float(value[i]) for key,value in errors.items()}) for i in range(len(x))])
    np.savez_compressed(output/'coefficients.npz', gamma=metrics['gamma'], beta=metrics['beta'])
    report = dict(checkpoint_epoch=checkpoint['epoch'], checkpoint=str(run_dir/'best.pt'), seed=args.seed,
        reference=reference_kind, b_min=low,b_max=high, environments=args.channels,
        environment_donors=donors.tolist(), coefficient_and_feature_statistics={key:stats(value) for key,value in metrics.items()},
        negative_gamma_fraction=float((metrics['gamma']<0).mean()), results=results,
        interpretation='Negative delta means this condition beats normal alpha=1. Intervals bootstrap independent environments; exploratory, not corrected for multiple comparisons. Shuffling diagnoses this one checkpoint, not benefit of retraining without geometry.')
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines = ['Modulation diagnostic', f'Checkpoint epoch: {checkpoint["epoch"]}',
        f'Fresh environments: {args.channels}; b/lambda: {low}..{high}', '',
        'Negative delta versus normal means BETTER than trained alpha=1.',
        'Intervals are paired 95% environment-bootstrap intervals (exploratory).', '']
    for r in results:
        lines.append(f'{r["condition"]}: {r["nmse_db"]:.3f} dB; delta {r["delta_vs_normal_db"]:+.3f} dB '
                     f'[{r["paired_bootstrap_delta_low"]:+.3f}, {r["paired_bootstrap_delta_high"]:+.3f}]')
    lines += ['', 'Feature change ratio: '+str(stats(metrics['change_ratio'])), '',
        'Decision guide:',
        '- If alpha=0.5 improves with its delta interval below zero, weaker modulation helps this held-out set.',
        '- If alpha=0 improves, bypassing modulation helps this checkpoint; it is not a retrained ablation.',
        '- If shuffled geometry worsens with its interval above zero, matching geometry contributes.',
        '- If shuffling changes little, geometry may be weakly used; similar pooled embeddings can also explain this.',
        '- Large gamma/beta or feature change alone does not prove harmful modulation.',
        '- Do not select settings on this diagnostic set and then call it an untouched final test.']
    (output/'decision_report.txt').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines[:6+len(results)]),flush=True)
    print('Saved diagnostics:', output.resolve(),flush=True)


if __name__ == '__main__':
    main()
