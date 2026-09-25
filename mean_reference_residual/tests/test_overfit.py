"""Copy into tests/; run python test_overfit.py from that folder.

Requires the current channel, geometry, dataset, model and training modules.
Intentionally trains AND evaluates on the same fixed tiny dataset.
This is an optimization diagnostic, not a validation/generalization experiment.
"""
import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from dataset.generator import generate_dataset
from model.estimator import ChannelEstimator
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--device', default=None, help='Default: CUDA if available, otherwise CPU.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--required-drop-db', type=float, default=10.0,
                        help='Required NMSE improvement relative to initial model.')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'artifacts' / 'overfit_mean_reference')
    args = parser.parse_args()
    if args.epochs < 1 or args.learning_rate <= 0 or args.required_drop_db <= 0:
        parser.error('epochs, learning-rate and required-drop-db must be positive.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # Small convolutions often run faster with fewer CPU worker threads.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    wavelength = 3e8 / 28e9
    config = dict(n_h_b=5, n_v_b=5, d_x_b=wavelength/8, d_y_b=wavelength/8,
        n_h_u=5, n_v_u=5, d_x_u=wavelength/8, d_y_u=wavelength/8,
        fc=28e9, fs=1e5, n_subcarriers=3, n_deformations=8,
        n_channels=2, n_paths=3, snr_values=[20], b_min=0.2, b_max=0.2,
        n_pilots=1, seed=args.seed)
    # Generated ONCE: paths, geometry, and measurement noise remain fixed.
    observations, geometry, targets = generate_dataset(**config)
    expected_samples = config['n_channels'] * (config['n_subcarriers'] - 1)
    assert observations.shape == (expected_samples, 32, 25, 25)
    assert geometry.shape == (expected_samples, 8, 25, 25, 6)
    assert targets.shape == (expected_samples, 4, 25, 25)
    for array in (observations, geometry, targets):
        assert np.isfinite(array).all(), 'Generated data contains nonfinite values.'
    dataset = TensorDataset(*(torch.as_tensor(array, dtype=torch.float32)
                              for array in (observations, geometry, targets)))
    loader = DataLoader(dataset, batch_size=expected_samples, shuffle=False, num_workers=0)
    model = ChannelEstimator(n_deformations=8)
    trainer = Trainer(model, device=args.device, learning_rate=args.learning_rate,
                      weight_decay=0.0, gradient_clip=1.0, output_dir=args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f'Overfitting {len(dataset)} fixed samples on {trainer.device}.', flush=True)
    print('The SAME loader is intentionally used for training and evaluation.', flush=True)
    print('No generalization or out-of-range claims can be made from this test.', flush=True)
    initial = trainer.evaluate(loader)
    print(f'Initial model NMSE: {initial["nmse_db"]:.2f} dB; '
          f'mean-observation reference: {initial["reference_nmse_db"]:.2f} dB', flush=True)

    # Verify every trainable component receives finite, nonzero gradients.
    batch = [tensor.to(trainer.device) for tensor in next(iter(loader))]
    obs, geo, residual_target = batch
    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    predicted = trainer.model(obs, geo)
    true_h0 = trainer.model.reconstruction.reference(obs) + residual_target
    from training.trainer import nmse_per_sample
    nmse_per_sample(predicted, true_h0).mean().backward()
    gradient_report = {}
    for name in ('cnn', 'geometry_encoder', 'modulation', 'decoder'):
        component = getattr(trainer.model, name)
        gradients = [p.grad for p in component.parameters() if p.requires_grad]
        assert gradients and all(g is not None for g in gradients), f'Missing gradient in {name}'
        assert all(torch.isfinite(g).all().item() for g in gradients), f'Nonfinite gradient in {name}'
        gradient_sum = sum(g.abs().sum().item() for g in gradients)
        assert gradient_sum > 0, f'No gradient reaches {name}'
        gradient_report[name] = gradient_sum
    trainer.optimizer.zero_grad(set_to_none=True)
    # Trainer's "validation" metric here is deliberately the training set metric.
    trainer.fit(loader, loader, epochs=args.epochs)
    final = trainer.evaluate(loader)  # fit restores its best epoch.
    drop_db = initial['nmse_db'] - final['nmse_db']
    beats_reference = final['nmse'] < final['reference_nmse']
    passed = drop_db >= args.required_drop_db and beats_reference
    report = dict(passed=passed, purpose='memorize a fixed tiny dataset, not generalization',
        config=config, epochs=args.epochs, learning_rate=args.learning_rate,
        device=str(trainer.device), initial=initial, best=final,
        improvement_db=drop_db, required_improvement_db=args.required_drop_db,
        beats_reference=beats_reference, initial_gradient_absolute_sums=gradient_report)
    (args.output / 'overfit_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Best NMSE: {final["nmse_db"]:.2f} dB; improvement: {drop_db:.2f} dB', flush=True)
    print('Report and checkpoint:', args.output.resolve(), flush=True)
    if not passed:
        raise SystemExit('OVERFIT TARGET NOT REACHED. Inspect history.json and overfit_report.json. '
                         'This is a diagnostic threshold, not proof of a model bug. '
                         'Check whether loss is still decreasing before extending training.')
    print('SMALL OVERFIT TEST PASSED', flush=True)


if __name__ == '__main__':
    main()
