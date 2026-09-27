import numpy as np


def generate_path_parameters(n_paths, fs, rng):
    """Generate one shared wideband geometric propagation environment."""
    if n_paths < 1:
        raise ValueError("n_paths must be at least 1.")

    return {
        "aoa_az": rng.uniform(-np.pi, np.pi, n_paths),
        "aoa_el": rng.uniform(-np.pi / 2, np.pi / 2, n_paths),
        "aod_az": rng.uniform(-np.pi, np.pi, n_paths),
        "aod_el": rng.uniform(-np.pi / 2, np.pi / 2, n_paths),
        "gain": (
            rng.standard_normal(n_paths)
            + 1j * rng.standard_normal(n_paths)
        ) / np.sqrt(2 * n_paths),
        "delay": rng.uniform(0.0, 1.0 / fs, n_paths),
    }
