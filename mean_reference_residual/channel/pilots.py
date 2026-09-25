import numpy as np


def generate_multi_pilots(n_u, n_pilots, rng):
    """Generate unitary DFT-based pilot matrices [N_U, N_U, T_p]."""
    if n_pilots < 1:
        raise ValueError("n_pilots must be at least 1.")

    indices = np.arange(n_u)
    dft = np.exp(
        -2j * np.pi * indices[:, None] * indices[None, :] / n_u
    ) / np.sqrt(n_u)
    pilots = np.empty((n_u, n_u, n_pilots), dtype=np.complex128)

    for pilot_index in range(n_pilots):
        phase = np.exp(2j * np.pi * rng.random(n_u))
        pilots[:, :, pilot_index] = dft @ np.diag(phase)

    return pilots


def transmit_pilots(channel, pilots, snr_db, rng):
    """Apply Y[k,t] = H[k]S[t] + N and return Y plus noise variance."""
    clean = np.einsum("buk,uvt->bvkt", channel, pilots)
    variance = np.mean(np.abs(clean) ** 2) / (10 ** (snr_db / 10))
    noise = np.sqrt(variance / 2) * (
        rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape)
    )
    return clean + noise, float(variance)


def ridge_estimate(observations, pilots, regularization):
    """Estimate one subcarrier channel from all pilot observations."""
    if observations.ndim != 3:
        raise ValueError("observations must have shape [N_B, N_U, T_p].")

    y_stack = np.concatenate(
        [observations[:, :, t] for t in range(observations.shape[2])],
        axis=1,
    )
    s_stack = np.concatenate(
        [pilots[:, :, t] for t in range(pilots.shape[2])],
        axis=1,
    )
    gram = s_stack @ s_stack.conj().T
    identity = np.eye(gram.shape[0], dtype=gram.dtype)
    return (
        y_stack
        @ s_stack.conj().T
        @ np.linalg.inv(gram + regularization * identity)
    )
