import numpy as np


def steering_vector_fim(rigid_positions, deformation, wavelength, azimuth, elevation):
    """Return the normalized steering vector of a deformed array."""
    rigid_positions = np.asarray(rigid_positions, dtype=float)
    deformation = np.asarray(deformation, dtype=float)

    if rigid_positions.ndim != 2 or rigid_positions.shape[0] != 3:
        raise ValueError("rigid_positions must have shape [3, N].")
    if deformation.shape != rigid_positions.shape:
        raise ValueError("deformation must have the same shape as rigid_positions.")

    direction = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    phase = (2 * np.pi / wavelength) * (
        direction @ (rigid_positions + deformation)
    )
    return np.exp(1j * phase) / np.sqrt(rigid_positions.shape[1])


def build_channel(
    path_parameters,
    p_b,
    z_b,
    p_u,
    z_u,
    wavelength,
    fs,
    n_subcarriers,
):
    """Build H with shape [N_B, N_U, K] for one physical environment."""
    n_b = p_b.shape[1]
    n_u = p_u.shape[1]
    channel = np.zeros((n_b, n_u, n_subcarriers), dtype=np.complex128)

    for path_index, gain in enumerate(path_parameters["gain"]):
        a_b = steering_vector_fim(
            p_b,
            z_b,
            wavelength,
            path_parameters["aoa_az"][path_index],
            path_parameters["aoa_el"][path_index],
        )
        a_u = steering_vector_fim(
            p_u,
            z_u,
            wavelength,
            path_parameters["aod_az"][path_index],
            path_parameters["aod_el"][path_index],
        )
        spatial_response = np.outer(a_b, np.conj(a_u))

        for k in range(n_subcarriers):
            frequency_offset = (k / n_subcarriers) * fs
            delay_phase = np.exp(
                -1j
                * 2
                * np.pi
                * frequency_offset
                * path_parameters["delay"][path_index]
            )
            channel[:, :, k] += gain * spatial_response * delay_phase

    return channel
