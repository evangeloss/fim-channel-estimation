"""Copy into training/trainer.py; use an empty training/__init__.py.

Loaders must yield (observations, geometry, residual_targets), all normalized
using the same observation-derived scale. Split physical channel realizations
BEFORE generating adjacent-subcarrier samples. No auxiliary loss is used.
"""
import json
import math
from pathlib import Path

import torch


def nmse_per_sample(prediction, target, epsilon=1e-12):
    """Linear per-sample NMSE across both complex subcarriers and antenna pairs."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError('Prediction and target must share shape [B,4,N_B,N_U].')
    error_power = (prediction - target).square().sum(dim=(1, 2, 3))
    target_power = target.square().sum(dim=(1, 2, 3))
    return error_power / target_power.clamp_min(epsilon)


def nmse_db(linear_mean):
    """dB of mean linear NMSE, not mean of individual dB values."""
    return 10 * math.log10(max(linear_mean, 1e-30))


class Trainer:
    """Jointly optimize all estimator components using AdamW.

    Saves best.pt (best validation epoch) and history.json in output_dir.
    A completed fit restores best validation weights and matching optimizer
    state. Calling fit again continues from that best checkpoint; it starts
    a new run history. Use a separate output directory for separate experiments.
    """

    def __init__(self, model, device=None, learning_rate=3e-4,
                 weight_decay=1e-4, gradient_clip=1.0, output_dir='artifacts'):
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError('learning_rate must be positive and weight_decay nonnegative.')
        if gradient_clip is not None and gradient_clip <= 0:
            raise ValueError('gradient_clip must be positive or None.')
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = model.to(device=self.device, dtype=torch.float32)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate,
                                          weight_decay=weight_decay)
        self.gradient_clip = gradient_clip
        self.output_dir = Path(output_dir)
        self.best_checkpoint = self.output_dir / 'best.pt'
        self.settings = dict(learning_rate=learning_rate, weight_decay=weight_decay,
                             gradient_clip=gradient_clip, device=str(self.device))

    def _run_epoch(self, loader, training):
        self.model.train(training)
        total_nmse = total_reference = 0.0
        sample_count = 0
        for batch in loader:
            if len(batch) != 3:
                raise ValueError('Each batch must contain observations, geometry, residual_targets.')
            observations, geometry, residual_targets = [
                tensor.to(device=self.device, dtype=torch.float32) for tensor in batch
            ]
            if observations.ndim != 4 or observations.shape[1] < 4:
                raise ValueError('Observations must be [B,4*M,N_B,N_U].')
            reference = self.model.reconstruction.reference(observations)
            if residual_targets.shape != reference.shape:
                raise ValueError('Residual target must be [B,4,N_B,N_U].')
            if not all(torch.isfinite(tensor).all().item()
                       for tensor in (observations, geometry, residual_targets)):
                raise ValueError('Batch contains nonfinite values.')
            true_h0 = reference + residual_targets
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                prediction = self.model(observations, geometry)
                sample_nmse = nmse_per_sample(prediction, true_h0)
                loss = sample_nmse.mean()
                if not torch.isfinite(loss).item():
                    raise FloatingPointError('Nonfinite NMSE; check data and model outputs.')
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                        self.gradient_clip if self.gradient_clip is not None else float('inf'),
                        error_if_nonfinite=True)
                    self.optimizer.step()
            total_nmse += sample_nmse.detach().sum().item()
            total_reference += nmse_per_sample(reference, true_h0).sum().item()
            sample_count += observations.shape[0]
        if sample_count == 0:
            raise ValueError('Loader yielded no samples.')
        mean = total_nmse / sample_count
        reference_mean = total_reference / sample_count
        return dict(nmse=mean, nmse_db=nmse_db(mean), reference_nmse=reference_mean,
                    reference_nmse_db=nmse_db(reference_mean), samples=sample_count)

    def evaluate(self, loader):
        """Evaluate held-out data without updates; preserves previous model mode."""
        was_training = self.model.training
        try:
            return self._run_epoch(loader, training=False)
        finally:
            self.model.train(was_training)

    def fit(self, train_loader, validation_loader, epochs=30):
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
            raise ValueError('epochs must be a positive integer.')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        history, best_nmse = [], float('inf')
        for epoch in range(1, epochs + 1):
            train = self._run_epoch(train_loader, training=True)
            validation = self.evaluate(validation_loader)
            history.append(dict(epoch=epoch, train=train, validation=validation))
            if validation['nmse'] < best_nmse:
                best_nmse = validation['nmse']
                checkpoint = dict(epoch=epoch, model_state_dict=self.model.state_dict(),
                    optimizer_state_dict=self.optimizer.state_dict(),
                    model_config=getattr(self.model, 'config', {}),
                    trainer_settings=self.settings, validation=validation)
                temporary = self.output_dir / 'best.tmp.pt'
                torch.save(checkpoint, temporary)
                temporary.replace(self.best_checkpoint)
            (self.output_dir / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
            print(f'Epoch {epoch:03d}/{epochs}: train={train["nmse_db"]:.2f} dB, '
                  f'validation={validation["nmse_db"]:.2f} dB, '
                  f'reference={validation["reference_nmse_db"]:.2f} dB', flush=True)
        checkpoint = torch.load(self.best_checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.eval()
        print(f'Restored best validation checkpoint: epoch {checkpoint["epoch"]}', flush=True)
        return history
