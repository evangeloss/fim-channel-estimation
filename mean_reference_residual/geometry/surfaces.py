import numpy as np


def _rotation_matrix(azimuth, elevation, roll):
    rz = np.array(
        [
            [np.cos(azimuth), -np.sin(azimuth), 0.0],
            [np.sin(azimuth), np.cos(azimuth), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    ry = np.array(
        [
            [np.cos(elevation), 0.0, np.sin(elevation)],
            [0.0, 1.0, 0.0],
            [-np.sin(elevation), 0.0, np.cos(elevation)],
        ]
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll), np.cos(roll)],
        ]
    )
    return rz @ ry @ rx


def _rigid_surface(n_h, n_v, d_x, d_y, center, azimuth, elevation, roll):
    rotation = _rotation_matrix(azimuth, elevation, roll)
    i_axis = rotation[:, 0]
    j_axis = rotation[:, 1]
    normal = rotation[:, 2]
    center = np.asarray(center, dtype=float).reshape(3)

    x_local = (np.arange(n_h) - (n_h - 1) / 2) * d_x
    y_local = (np.arange(n_v) - (n_v - 1) / 2) * d_y
    positions = np.stack(
        [
            center + x_local[h] * i_axis + y_local[v] * j_axis
            for v in range(n_v)
            for h in range(n_h)
        ],
        axis=1,
    )
    return positions, normal


def generate_reference_geometry(
    n_h_b,
    n_v_b,
    d_x_b,
    d_y_b,
    n_h_u,
    n_v_u,
    d_x_u,
    d_y_u,
    wavelength,
):
    """Return fixed facing BS/UE rigid arrays and their surface normals."""
    p_b, normal_b = _rigid_surface(
        n_h_b,
        n_v_b,
        d_x_b,
        d_y_b,
        center=[0.0, 0.0, 0.0],
        azimuth=0.0,
        elevation=np.pi / 2,
        roll=0.0,
    )
    p_u, normal_u = _rigid_surface(
        n_h_u,
        n_v_u,
        d_x_u,
        d_y_u,
        center=[5 * wavelength, 0.0, 0.0],
        azimuth=0.0,
        elevation=-np.pi / 2,
        roll=0.0,
    )
    return p_b, p_u, normal_b, normal_u


def generate_deformation_codebook(
    p_b,
    p_u,
    normal_b,
    normal_u,
    wavelength,
    b_over_lambda,
    n_deformations,
    rng,
):
    """Generate M random normal-displacement geometries for one sample."""
    n_b = p_b.shape[1]
    n_u = p_u.shape[1]
    amplitude = b_over_lambda * wavelength

    p_b_all = []
    z_b_all = []
    p_u_all = []
    z_u_all = []

    for _ in range(n_deformations):
        xi_b = rng.uniform(-amplitude, amplitude, n_b)
        xi_u = rng.uniform(-amplitude, amplitude, n_u)
        p_b_all.append(p_b.copy())
        p_u_all.append(p_u.copy())
        z_b_all.append(normal_b[:, None] * xi_b[None, :])
        z_u_all.append(normal_u[:, None] * xi_u[None, :])

    return p_b_all, z_b_all, p_u_all, z_u_all
