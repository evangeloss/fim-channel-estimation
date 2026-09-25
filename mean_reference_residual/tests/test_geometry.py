"""Run from the project root: python test_dataset.py

Requires your channel/, geometry/, and dataset/ packages and NumPy.
No training or PyTorch model is involved. This checks the real generator
against captured channel, geometry, and tentative-estimate intermediates.
"""
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import dataset.generator as generator
from unittest.mock import patch
import numpy as np


def check_case(b, seed=42):
    wavelength = 3e8 / 28e9
    config = dict(
        n_h_b=2, n_v_b=2, d_x_b=wavelength / 8, d_y_b=wavelength / 8,
        n_h_u=2, n_v_u=2, d_x_u=wavelength / 8, d_y_u=wavelength / 8,
        fc=28e9, fs=1e5, n_subcarriers=3, n_deformations=3,
        n_channels=2, n_paths=3, snr_values=[10],
        b_min=b, b_max=b, n_pilots=1, seed=seed,
    )
    channel_records, estimate_records, geometry_records = [], [], []
    reference_records = []
    original_channel = generator.build_channel
    original_estimate = generator.ridge_estimate
    original_codebook = generator.generate_deformation_codebook
    original_reference = generator.generate_reference_geometry

    def capture_channel(*args, **kwargs):
        result = original_channel(*args, **kwargs)
        channel_records.append(result.copy())
        return result

    def capture_estimate(*args, **kwargs):
        result = original_estimate(*args, **kwargs)
        estimate_records.append(result.copy())
        return result

    def capture_codebook(*args, **kwargs):
        result = original_codebook(*args, **kwargs)
        geometry_records.append(tuple([x.copy() for x in group] for group in result))
        return result

    def capture_reference(*args, **kwargs):
        result = original_reference(*args, **kwargs)
        reference_records.append(tuple(x.copy() for x in result))
        return result

    with patch.object(generator, 'build_channel', side_effect=capture_channel), \
         patch.object(generator, 'ridge_estimate', side_effect=capture_estimate), \
         patch.object(generator, 'generate_deformation_codebook', side_effect=capture_codebook), \
         patch.object(generator, 'generate_reference_geometry', side_effect=capture_reference):
        observations, geometry, targets = generator.generate_dataset(**config)

    count, m_count, k_count, nb, nu = 2, 3, 3, 4, 4
    samples = count * (k_count - 1)
    assert observations.shape == (samples, 4 * m_count, nb, nu)
    assert geometry.shape == (samples, m_count, nb, nu, 6)
    assert targets.shape == (samples, 4, nb, nu)
    for array in (observations, geometry, targets):
        assert array.dtype == np.float32
        assert np.isfinite(array).all()

    assert len(channel_records) == count * (m_count + 1)
    assert len(estimate_records) == count * m_count * k_count
    origin = reference_records[0][0].mean(axis=1)
    np.testing.assert_allclose(origin, np.zeros(3), atol=1e-12)
    packed = observations.reshape(samples, m_count, 4, nb, nu)
    complex_observations = np.stack([
        packed[:, :, 0] + 1j * packed[:, :, 1],
        packed[:, :, 2] + 1j * packed[:, :, 3],
    ], axis=-1)
    assert np.max(np.abs(complex_observations)) <= 1.0 + 2e-6
    reconstructed = observations.reshape(samples, m_count, 4, nb, nu).mean(axis=1) + targets
    reconstructed_complex = np.stack([
        reconstructed[:, 0] + 1j * reconstructed[:, 1],
        reconstructed[:, 2] + 1j * reconstructed[:, 3],
    ], axis=-1)

    for realization in range(count):
        true_channel = channel_records[realization * (m_count + 1)]
        estimates = []
        for m in range(m_count):
            offset = (realization * m_count + m) * k_count
            estimates.append(np.stack(estimate_records[offset:offset + k_count], axis=-1))
            if b == 0:
                np.testing.assert_allclose(
                    channel_records[realization * (m_count + 1) + m + 1],
                    true_channel, rtol=1e-10, atol=1e-10,
                )
        pb, zb, pu, zu = geometry_records[realization]
        for k in range(k_count - 1):
            sample = realization * (k_count - 1) + k
            scale = max(float(np.max(np.abs(e[:, :, k:k + 2]))) for e in estimates) + 1e-8
            np.testing.assert_allclose(
                reconstructed_complex[sample], true_channel[:, :, k:k + 2] / scale,
                rtol=2e-5, atol=2e-6,
                err_msg='Residual plus mean observation must recover normalized H0.',
            )
            for m in range(m_count):
                np.testing.assert_allclose(
                    complex_observations[sample, m], estimates[m][:, :, k:k + 2] / scale,
                    rtol=2e-5, atol=2e-6,
                    err_msg='Observation ordering or observation-only normalization mismatch.',
                )
                bs = ((pb[m] + zb[m] - origin[:, None]) / wavelength).T
                ue = ((pu[m] + zu[m] - origin[:, None]) / wavelength).T
                np.testing.assert_allclose(geometry[sample, m, :, :, :3],
                    np.broadcast_to(bs[:, None, :], (nb, nu, 3)), atol=2e-6)
                np.testing.assert_allclose(geometry[sample, m, :, :, 3:],
                    np.broadcast_to(ue[None, :, :], (nb, nu, 3)), atol=2e-6)

    # The same seed should reproduce the complete sample, including noise.
    repeated = generator.generate_dataset(**config)
    for actual, expected in zip(repeated, (observations, geometry, targets)):
        np.testing.assert_array_equal(actual, expected)
    print(f'PASS: b/lambda={b}, observations={observations.shape}, geometry={geometry.shape}, targets={targets.shape}')


if __name__ == '__main__':
    check_case(0.0)
    check_case(0.2)
    print('DATASET SMOKE TEST PASSED')
    print('Checked assembly and consistency; this does not validate the physical channel model.')
