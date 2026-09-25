"""Copy to main/train.py and create an empty main/__init__.py.

From the project root: python -m main.train
From main/: python train.py
Quick pipeline check: python train.py --quick

Requires the previously supplied dataset generator, estimator and trainer.
"""
import argparse
import csv
import json
import random
import sys
from datetime import datetime
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
    parser.add_argument('--quick', action='store_true', help='Use 8/4/4 realizations and 3 epochs; wiring check only.')
    parser.add_argument('--train-channels', type=int, default=200)
    parser.add_argument('--validation-channels', type=int, default=50)
    parser.add_argument('--test-channels', type=int, default=50)
    parser.add_argument('--subcarriers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--snr-db', type=float, default=20.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default=None)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    if args.quick:
        args.train_channels, args.validation_channels, args.test_channels, args.epochs = 8, 4, 4, 3
    if min(args.train_channels, args.validation_channels, args.test_channels, args.epochs, args.batch_size) < 1:
        parser.error('Sample counts, epochs, and batch size must be positive.')
    if args.subcarriers < 2 or args.learning_rate <= 0 or args.seed < 0 or not np.isfinite(args.snr_db):
        parser.error('Require subcarriers >= 2, positive learning rate, nonnegative seed, and finite SNR.')
    output = args.output or PROJECT_ROOT / 'artifacts' / datetime.now().strftime('training_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ('experiment_config.json', 'best.pt', 'history.json')):
        parser.error(f'{output} already contains a run. Choose a new --output folder.')

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    wl = 3e8 / 28e9
    simulation = dict(n_h_b=5, n_v_b=5, d_x_b=wl/8, d_y_b=wl/8,
        n_h_u=5, n_v_u=5, d_x_u=wl/8, d_y_u=wl/8,
        fc=28e9, fs=1e5, n_subcarriers=args.subcarriers,
        n_deformations=8, n_paths=3, n_pilots=1, snr_values=[args.snr_db], residual_reference='mean')
    splits = dict(
    train=dict(n_channels=args.train_channels, b_min=0.05, b_max=0.05, seed=args.seed+101),
    validation=dict(n_channels=args.validation_channels, b_min=0.05, b_max=0.05, seed=args.seed+202),
    extrapolation=dict(n_channels=args.test_channels, b_min=0.35, b_max=0.50, seed=args.seed+303))
    estimated_bytes = sum(s['n_channels'] for s in splits.values()) * (args.subcarriers-1) * 25*25*(32+8*6+4)*4
    config = dict(simulation=simulation, splits=splits, seed=args.seed,
        training=dict(epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, weight_decay=1e-4, gradient_clip=1.0),
        quick=args.quick, estimated_dataset_bytes=estimated_bytes,
        checkpoint_selection='validation only; extrapolation evaluated after training',
        target='normalized antenna-domain H0 = mean observation + dataset residual',
        metric='10*log10(mean per-sample linear NMSE), both adjacent subcarriers together')
    (output / 'experiment_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'Output: {output.resolve()}', flush=True)
    print(f'Approximate dataset tensor storage: {estimated_bytes/1024**2:.1f} MiB (excludes model/workspace).', flush=True)
    print('Each split uses independent physical realizations; no random split of subcarrier pairs.', flush=True)
    if args.quick:
        print('QUICK MODE: pipeline check, not a performance experiment.', flush=True)

    def make_loader(name):
        spec = splits[name]
        count = spec['n_channels'] * (args.subcarriers-1)
        print(f'Generating {name}: {spec["n_channels"]} realizations, {count} samples...', flush=True)
        arrays = generate_dataset(**simulation, **spec)
        expected_shapes = [(count, 32, 25, 25), (count, 8, 25, 25, 6), (count, 4, 25, 25)]
        for array, expected in zip(arrays, expected_shapes):
            if array.shape != expected or not np.isfinite(array).all():
                raise ValueError(f'Invalid {name} dataset; expected finite array of shape {expected}.')
        dataset = TensorDataset(*(torch.as_tensor(a, dtype=torch.float32) for a in arrays))
        shuffle_rng = torch.Generator().manual_seed(args.seed+404)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=(name == 'train'),
                          num_workers=0, drop_last=False, generator=shuffle_rng)

    train_loader = make_loader('train')
    validation_loader = make_loader('validation')
    # Initialize a fresh model. The overfit checkpoint is never loaded.
    model = ChannelEstimator(n_deformations=8, residual_reference='mean')
    trainer = Trainer(model, device=args.device, learning_rate=args.learning_rate,
        weight_decay=1e-4, gradient_clip=1.0, output_dir=output)
    config['model'] = model.config
    config['device'] = str(trainer.device)
    config['torch_version'] = str(torch.__version__)
    (output / 'experiment_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'Training on {trainer.device}', flush=True)
    trainer.fit(train_loader, validation_loader, epochs=args.epochs)
    # fit restores the best VALIDATION checkpoint. Only now expose the test set.
    validation_result = trainer.evaluate(validation_loader)
    del train_loader, validation_loader
    extrapolation_loader = make_loader('extrapolation')
    extrapolation_result = trainer.evaluate(extrapolation_loader)
    results = dict(validation=validation_result, extrapolation=extrapolation_result)
    for result in results.values():
        result['improvement_over_reference_db'] = result['reference_nmse_db'] - result['nmse_db']
    (output / 'evaluation_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    with (output / 'evaluation_results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['split'] + list(validation_result))
        writer.writeheader()
        for name, result in results.items():
            writer.writerow(dict(split=name, **result))
    print('\nBest-validation-checkpoint results:', flush=True)
    for name, result in results.items():
        print(f'{name}: model {result["nmse_db"]:.2f} dB; reference '
              f'{result["reference_nmse_db"]:.2f} dB; improvement '
              f'{result["improvement_over_reference_db"]:.2f} dB', flush=True)
    print('Saved best.pt, history.json, experiment_config.json, and evaluation_results.json/csv.', flush=True)
    print('Validation selected the checkpoint; it is not an untouched in-range test set.', flush=True)


if __name__ == '__main__':
    main()
