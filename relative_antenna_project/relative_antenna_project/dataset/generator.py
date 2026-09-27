import numpy as np

from channel import (
    build_channel,
    generate_multi_pilots,
    generate_path_parameters,
    ridge_estimate,
    transmit_pilots,
)
from geometry import (
    build_fixed_origin_features,
    generate_deformation_codebook,
    generate_reference_geometry,
)


def generate_dataset(
    n_h_b,
    n_v_b,
    d_x_b,
    d_y_b,
    n_h_u,
    n_v_u,
    d_x_u,
    d_y_u,
    fc,
    fs,
    n_subcarriers,
    n_deformations,
    n_channels,
    n_paths,
    snr_values,
    b_min=0.05,
    b_max=0.5,
    n_pilots=1,
    seed=None,
    residual_reference='mean',
):
    """Generate observation, geometry, and residual-target tensors.

    Returns
    -------
    observations : float32 [samples, 4*M, N_B, N_U]
    geometry : float32 [samples, M, N_B, N_U, 6]
    targets : float32 [samples, 4, N_B, N_U]
    """
    if n_subcarriers < 2:
        raise ValueError("At least two subcarriers are required.")
    if not 0 <= b_min <= b_max:
        raise ValueError("Require 0 <= b_min <= b_max.")

    if residual_reference not in ('mean', 'first'):
        raise ValueError('residual_reference must be mean or first.')
    rng = np.random.default_rng(seed)
    wavelength = 3e8 / fc
    n_b = n_h_b * n_v_b
    n_u = n_h_u * n_v_u
    samples_per_channel = n_subcarriers - 1
    n_samples = n_channels * samples_per_channel
    snr_values = np.asarray(snr_values, dtype=float).reshape(-1)
    if snr_values.size == 0:
        raise ValueError("snr_values cannot be empty.")

    observations = np.empty(
        (n_samples, 4 * n_deformations, n_b, n_u), dtype=np.float32
    )
    geometry = np.empty(
        (n_samples, n_deformations, n_b, n_u, 6), dtype=np.float32
    )
    targets = np.empty((n_samples, 4, n_b, n_u), dtype=np.float32)

    p_b, p_u, normal_b, normal_u = generate_reference_geometry(
        n_h_b,
        n_v_b,
        d_x_b,
        d_y_b,
        n_h_u,
        n_v_u,
        d_x_u,
        d_y_u,
        wavelength,
    )
    zero_b = np.zeros_like(p_b)
    zero_u = np.zeros_like(p_u)
    pilots = generate_multi_pilots(n_u, n_pilots, rng)
    origin = p_b.mean(axis=1)

    sample_index = 0
    for _ in range(n_channels):
        b_over_lambda = float(rng.uniform(b_min, b_max))
        snr_db = float(rng.choice(snr_values))
        paths = generate_path_parameters(n_paths, fs, rng)
        true_channel = build_channel(
            paths, p_b, zero_b, p_u, zero_u, wavelength, fs, n_subcarriers
        )

        p_b_all, z_b_all, p_u_all, z_u_all = generate_deformation_codebook(
            p_b,
            p_u,
            normal_b,
            normal_u,
            wavelength,
            b_over_lambda,
            n_deformations,
            rng,
        )
        pair_geometry = build_fixed_origin_features(
            p_b_all,
            z_b_all,
            p_u_all,
            z_u_all,
            wavelength,
            origin,
        )

        estimates = []
        for m in range(n_deformations):
            deformed_channel = build_channel(
                paths,
                p_b_all[m],
                z_b_all[m],
                p_u_all[m],
                z_u_all[m],
                wavelength,
                fs,
                n_subcarriers,
            )
            noisy_observations, variance = transmit_pilots(
                deformed_channel, pilots, snr_db, rng
            )
            estimate = np.stack(
                [
                    ridge_estimate(
                        noisy_observations[:, :, k, :], pilots, variance
                    )
                    for k in range(n_subcarriers)
                ],
                axis=2,
            )
            estimates.append(estimate)

        for k in range(samples_per_channel):
            scale = max(
                float(np.max(np.abs(estimate[:, :, k : k + 2])))
                for estimate in estimates
            ) + 1e-8

            for m, estimate in enumerate(estimates):
                r0 = estimate[:, :, k] / scale
                r1 = estimate[:, :, k + 1] / scale
                channel_start = 4 * m
                observations[sample_index, channel_start + 0] = r0.real
                observations[sample_index, channel_start + 1] = r0.imag
                observations[sample_index, channel_start + 2] = r1.real
                observations[sample_index, channel_start + 3] = r1.imag

            reference = (np.mean(np.stack(estimates, axis=0), axis=0)
                         if residual_reference == 'mean' else estimates[0])
            reference_0 = reference[:, :, k] / scale
            reference_1 = reference[:, :, k + 1] / scale
            delta_0 = true_channel[:, :, k] / scale - reference_0
            delta_1 = true_channel[:, :, k + 1] / scale - reference_1
            targets[sample_index, 0] = delta_0.real
            targets[sample_index, 1] = delta_0.imag
            targets[sample_index, 2] = delta_1.real
            targets[sample_index, 3] = delta_1.imag
            geometry[sample_index] = pair_geometry
            sample_index += 1

    return observations, geometry, targets
