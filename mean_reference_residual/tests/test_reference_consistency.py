"""Run from tests/: python test_reference_consistency.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from dataset.generator import generate_dataset
from model.residual_reconstruction import ResidualReconstruction


def main():
    wl = 3e8/28e9
    settings = dict(n_h_b=2, n_v_b=2, d_x_b=wl/8, d_y_b=wl/8,
        n_h_u=2, n_v_u=2, d_x_u=wl/8, d_y_u=wl/8,
        fc=28e9, fs=1e5, n_subcarriers=3, n_deformations=3,
        n_channels=2, n_paths=3, snr_values=[20], seed=567)
    x, g, target_mean = generate_dataset(**settings, residual_reference='mean')
    x_old, g_old, target_first = generate_dataset(**settings, residual_reference='first')
    np.testing.assert_array_equal(x, x_old)
    np.testing.assert_array_equal(g, g_old)
    mean = x.reshape(len(x), 3, 4, 4, 4).mean(axis=1)
    np.testing.assert_allclose(mean+target_mean, x[:, :4]+target_first, atol=2e-6, rtol=2e-5)
    observations = torch.from_numpy(x)
    for kind, target in [('mean', target_mean), ('first', target_first)]:
        reconstruct = ResidualReconstruction(2, 2, 2, 2, residual_reference=kind)
        target = torch.from_numpy(target)
        reference = reconstruct.reference(observations)
        zero = reconstruct(torch.zeros_like(target), observations)
        torch.testing.assert_close(zero, reference, rtol=0, atol=0)
        actual = reconstruct(reconstruct.beamspace(target), observations)
        torch.testing.assert_close(actual, reference+target, rtol=2e-5, atol=2e-6)
    print('REFERENCE CONSISTENCY TEST PASSED')
    print('Inputs unchanged; both target conventions recover the same H0; zero correction uses the configured reference.')


if __name__ == '__main__':
    main()
