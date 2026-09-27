import numpy as np


def build_pair_geometry_features(p_b_all, z_b_all, p_u_all, z_u_all, wavelength):
    """Return pair descriptors with shape [M, N_B, N_U, 15]."""
    n_deformations = len(p_b_all)
    if not (
        len(z_b_all) == len(p_u_all) == len(z_u_all) == n_deformations
    ):
        raise ValueError("All geometry lists must have the same length.")
    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")

    feature_grids = []
    for m in range(n_deformations):
        p_b = np.asarray(p_b_all[m], dtype=np.float32)
        z_b = np.asarray(z_b_all[m], dtype=np.float32)
        p_u = np.asarray(p_u_all[m], dtype=np.float32)
        z_u = np.asarray(z_u_all[m], dtype=np.float32)

        if p_b.ndim != 2 or p_b.shape[0] != 3:
            raise ValueError("BS coordinates must have shape [3, N_B].")
        if p_u.ndim != 2 or p_u.shape[0] != 3:
            raise ValueError("UE coordinates must have shape [3, N_U].")
        if p_b.shape != z_b.shape or p_u.shape != z_u.shape:
            raise ValueError("Rigid positions and deformations must match.")

        n_b = p_b.shape[1]
        n_u = p_u.shape[1]
        local_b = (p_b - p_b.mean(axis=1, keepdims=True)) / wavelength
        local_u = (p_u - p_u.mean(axis=1, keepdims=True)) / wavelength
        deformation_b = z_b / wavelength
        deformation_u = z_u / wavelength
        position_b = (p_b + z_b) / wavelength
        position_u = (p_u + z_u) / wavelength
        relative_position = position_u.T[None, :, :] - position_b.T[:, None, :]

        descriptor = np.concatenate(
            [
                np.broadcast_to(local_b.T[:, None, :], (n_b, n_u, 3)),
                np.broadcast_to(local_u.T[None, :, :], (n_b, n_u, 3)),
                np.broadcast_to(deformation_b.T[:, None, :], (n_b, n_u, 3)),
                np.broadcast_to(deformation_u.T[None, :, :], (n_b, n_u, 3)),
                relative_position,
            ],
            axis=-1,
        )
        feature_grids.append(descriptor.astype(np.float32))

    return np.stack(feature_grids, axis=0)

def build_fixed_origin_features(p_b_all, z_b_all, p_u_all, z_u_all,
                                wavelength, origin):
    """Return [M, N_B, N_U, 6] actual-position descriptors.

    Order: BS x,y,z followed by UE x,y,z, all in wavelength units.
    The supplied origin is fixed across observations and samples.
    No explicit displacement or BS-to-UE difference is included.
    """
    count = len(p_b_all)
    if count == 0 or not all(len(x) == count for x in
                             (z_b_all, p_u_all, z_u_all)):
        raise ValueError("Geometry lists must have the same nonzero length.")
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("wavelength must be finite and positive.")
    origin = np.asarray(origin, dtype=np.float64).reshape(3, 1)
    grids = []
    for p_b, z_b, p_u, z_u in zip(p_b_all, z_b_all, p_u_all, z_u_all):
        p_b, z_b, p_u, z_u = [np.asarray(x, dtype=np.float64)
                              for x in (p_b, z_b, p_u, z_u)]
        for positions, displacement in ((p_b, z_b), (p_u, z_u)):
            if positions.ndim != 2 or positions.shape[0] != 3:
                raise ValueError("Positions must have shape [3, N].")
            if displacement.shape != positions.shape:
                raise ValueError("Displacement must match positions.")
        bs = ((p_b + z_b - origin) / wavelength).T
        ue = ((p_u + z_u - origin) / wavelength).T
        shape = (len(bs), len(ue), 3)
        grids.append(np.concatenate([
            np.broadcast_to(bs[:, None, :], shape),
            np.broadcast_to(ue[None, :, :], shape),
        ], axis=-1))
    features = np.stack(grids).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Geometry contains non-finite coordinates.")
    return features
