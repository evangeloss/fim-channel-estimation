
"""
test_representation_quality_vs_b.py

Diagnostics for FIM morphing-range generalization.

This script compares, versus morphing range b/lambda:

1) Fixed 2-D UPA DFT representation
2) Geometry-aware steering-dictionary representation
3) Oracle geometry-aware path basis (sanity upper bound)

Main diagnostics:
-----------------
A. Top-K reconstruction NMSE
   - DFT: keep K strongest DFT coefficients, reconstruct H
   - Geometry-aware dictionary: keep K strongest LS coefficients, reconstruct H
   - Oracle basis: reconstruct from the exact simulated path directions

B. Geometry-aware dictionary mutual coherence
       mu(A) = max_{i != j} |a_i^H a_j| / (||a_i|| ||a_j||)

C. Geometry-aware dictionary condition number
       kappa(A)

D. Optional coefficient concentration C_K
   Included mainly for continuity with previous tests.

Interpretation:
---------------
If morphing causes
    DFT reconstruction NMSE      -> worse with b
while
    geometry-aware NMSE          -> stable / better
and
    dictionary coherence/cond    -> improve or remain controlled,

then the main problem is likely representation mismatch rather than
insufficient CNN depth.

IMPORTANT:
----------
For paper-quality results, replace the stand-alone deformation and channel
functions below with the exact functions used by your dataset/training code
(e.g. your actual deformation codebook and build_H_fim_from_paths logic).

The experimental protocol should stay the same:
- Same physical paths across all b values within one realization.
- Same normalized deformation shapes across all b values within one realization.
- Only deformation amplitude b changes.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

c = 3e8
fc = 28e9
wavelength = c / fc

# Array sizes
NH_B = 5
NV_B = 5
NH_U = 5
NV_U = 5

N_B = NH_B * NV_B
N_U = NH_U * NV_U

# Element spacing
dx_B = wavelength / 8
dy_B = wavelength / 8
dx_U = wavelength / 8
dy_U = wavelength / 8

# Channel
L_paths = 3

# Deformation observations
M = 8

# Monte-Carlo realizations
Nreal = 200

# Morphing range sweep
b_list = np.array([
    0.00,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
])

# Top-K coefficients retained in reconstruction
K_list = [1, 3, 5, 10, 25]

# Geometry-aware angular dictionary grid
# 5 x 5 = 25 atoms, matching array dimension for a clean first comparison.
N_AZ = 5
N_EL = 5

AZ_MIN = -np.pi / 2
AZ_MAX = +np.pi / 2

EL_MIN = -np.pi / 3
EL_MAX = +np.pi / 3

# Regularization for LS / pseudo-inverse
REGULARIZATION = 1e-4

OUTPUT_DIR = Path("representation_quality_vs_b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. ARRAY GEOMETRY
# ============================================================

def create_upa_positions(NH, NV, dx, dy):
    x = (np.arange(NH) - (NH - 1) / 2) * dx
    y = (np.arange(NV) - (NV - 1) / 2) * dy

    X, Y = np.meshgrid(x, y, indexing="ij")

    return np.column_stack([
        X.reshape(-1),
        Y.reshape(-1),
        np.zeros(NH * NV)
    ])


P_B0 = create_upa_positions(NH_B, NV_B, dx_B, dy_B)
P_U0 = create_upa_positions(NH_U, NV_U, dx_U, dy_U)


# ============================================================
# 3. FIXED 2-D UPA DFT
# ============================================================

def unitary_dft(N):
    n = np.arange(N)
    k = n[:, None]

    return (
        np.exp(-1j * 2 * np.pi * k * n / N)
        / np.sqrt(N)
    )


def upa_dft(NH, NV):
    F_H = unitary_dft(NH)
    F_V = unitary_dft(NV)

    # Matches create_upa_positions(...).reshape(-1)
    return np.kron(F_H, F_V)


F_B = upa_dft(NH_B, NV_B)
F_U = upa_dft(NH_U, NV_U)


def fixed_dft_coefficients(H):
    """
    Unitary transform:
        S_DFT = F_B^H H F_U
    """
    return F_B.conj().T @ H @ F_U


def reconstruct_from_dft(S):
    """
        H_hat = F_B S F_U^H
    """
    return F_B @ S @ F_U.conj().T


# ============================================================
# 4. PHYSICAL CHANNEL MODEL
# ============================================================

def direction_vector(azimuth, elevation):
    return np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation)
    ])


def steering_vector(positions, direction):
    phase = (
        2 * np.pi / wavelength
        * positions @ direction
    )

    a = np.exp(1j * phase)

    return a / np.sqrt(len(a))


def generate_path_parameters(rng):
    """
    One physical multipath realization.
    Angles and gains are fixed across all b in that realization.
    """
    paths = []

    for _ in range(L_paths):
        az_B = rng.uniform(AZ_MIN, AZ_MAX)
        el_B = rng.uniform(EL_MIN, EL_MAX)

        az_U = rng.uniform(AZ_MIN, AZ_MAX)
        el_U = rng.uniform(EL_MIN, EL_MAX)

        u_B = direction_vector(az_B, el_B)
        u_U = direction_vector(az_U, el_U)

        alpha = (
            rng.normal() + 1j * rng.normal()
        ) / np.sqrt(2 * L_paths)

        paths.append({
            "alpha": alpha,
            "u_B": u_B,
            "u_U": u_U,
            "az_B": az_B,
            "el_B": el_B,
            "az_U": az_U,
            "el_U": el_U,
        })

    return paths


def build_channel(P_B, P_U, paths):
    H = np.zeros((N_B, N_U), dtype=np.complex128)

    for path in paths:
        a_B = steering_vector(P_B, path["u_B"])
        a_U = steering_vector(P_U, path["u_U"])

        H += (
            path["alpha"]
            * np.outer(a_B, np.conj(a_U))
        )

    return H


# ============================================================
# 5. DEFORMATION MODEL
# ============================================================

def normalize_profile(z):
    z = z - np.mean(z)

    m = np.max(np.abs(z))
    if m > 1e-12:
        z = z / m

    return z


def generate_single_profile(NH, NV, rng):
    """
    Example smooth deformation generator.
    Replace with your exact codebook generator for final experiments.
    """
    x = np.linspace(-1, 1, NH)
    y = np.linspace(-1, 1, NV)

    X, Y = np.meshgrid(x, y, indexing="ij")

    deformation_type = rng.integers(0, 4)

    if deformation_type == 0:
        profile = X**2 + 0.7 * Y**2

    elif deformation_type == 1:
        phase = rng.uniform(0, 2 * np.pi)
        profile = (
            np.sin(np.pi * X + phase)
            * np.cos(np.pi * Y)
        )

    elif deformation_type == 2:
        ax = rng.uniform(-1, 1)
        ay = rng.uniform(-1, 1)
        profile = ax * X + ay * Y

    else:
        a = rng.uniform(-1, 1)
        b = rng.uniform(-1, 1)
        d = rng.uniform(-1, 1)

        profile = (
            a * X
            + b * Y
            + d * np.sin(np.pi * X) * np.cos(np.pi * Y)
        )

    return normalize_profile(profile.reshape(-1))


def generate_deformation_codebook(rng):
    profiles_B = []
    profiles_U = []

    for _ in range(M):
        profiles_B.append(
            generate_single_profile(NH_B, NV_B, rng)
        )
        profiles_U.append(
            generate_single_profile(NH_U, NV_U, rng)
        )

    return (
        np.stack(profiles_B),
        np.stack(profiles_U),
    )


def apply_deformation(P0, profile, b_over_lambda):
    P = P0.copy()

    dz = (
        b_over_lambda
        * wavelength
        * profile
    )

    P[:, 2] += dz

    return P


# ============================================================
# 6. GEOMETRY-AWARE STEERING DICTIONARY
# ============================================================

azimuth_grid = np.linspace(
    AZ_MIN,
    AZ_MAX,
    N_AZ
)

elevation_grid = np.linspace(
    EL_MIN,
    EL_MAX,
    N_EL
)


def build_geometry_dictionary(positions):
    """
    Dictionary columns use the ACTUAL deformed coordinates.
    """
    columns = []

    for az in azimuth_grid:
        for el in elevation_grid:
            u = direction_vector(az, el)
            columns.append(
                steering_vector(positions, u)
            )

    return np.column_stack(columns)


def regularized_left_inverse(A, reg=REGULARIZATION):
    """
    L = (A^H A + reg I)^(-1) A^H
    """
    G = A.conj().T @ A

    I = np.eye(
        G.shape[0],
        dtype=G.dtype
    )

    return np.linalg.solve(
        G + reg * I,
        A.conj().T
    )


def geometry_dictionary_coefficients(H, P_B, P_U):
    """
    H ≈ A_B S A_U^H

    S ≈ A_B^dagger H (A_U^dagger)^H
    """
    A_B = build_geometry_dictionary(P_B)
    A_U = build_geometry_dictionary(P_U)

    L_B = regularized_left_inverse(A_B)
    L_U = regularized_left_inverse(A_U)

    S = (
        L_B
        @ H
        @ L_U.conj().T
    )

    return S, A_B, A_U


def reconstruct_from_geometry_dictionary(S, A_B, A_U):
    return (
        A_B
        @ S
        @ A_U.conj().T
    )


# ============================================================
# 7. ORACLE PATH BASIS
# ============================================================

def oracle_path_reconstruction(P_B, P_U, paths):
    """
    Uses the TRUE simulated path directions.

    This is not a practical estimator.
    It is a sanity upper bound showing whether the physical channel
    remains low-dimensional under deformation when the correct geometry
    and exact path directions are known.
    """
    H_oracle = np.zeros(
        (N_B, N_U),
        dtype=np.complex128
    )

    for path in paths:
        a_B = steering_vector(
            P_B,
            path["u_B"]
        )

        a_U = steering_vector(
            P_U,
            path["u_U"]
        )

        H_oracle += (
            path["alpha"]
            * np.outer(
                a_B,
                np.conj(a_U)
            )
        )

    return H_oracle


# ============================================================
# 8. METRICS
# ============================================================

def nmse(reference, estimate):
    return (
        np.linalg.norm(
            reference - estimate,
            "fro"
        )**2
        /
        (
            np.linalg.norm(reference, "fro")**2
            + 1e-15
        )
    )


def to_db(x):
    return 10 * np.log10(
        np.maximum(x, 1e-15)
    )


def retain_top_k(X, K):
    """
    Keep only the K largest-magnitude coefficients.
    """
    flat = X.reshape(-1)
    K = min(K, flat.size)

    if K == flat.size:
        return X.copy()

    idx = np.argpartition(
        np.abs(flat),
        -K
    )[-K:]

    out = np.zeros_like(flat)
    out[idx] = flat[idx]

    return out.reshape(X.shape)


def top_k_concentration(X, K):
    energy = np.abs(X.reshape(-1))**2
    total = np.sum(energy)

    if total < 1e-15:
        return 0.0

    K = min(K, len(energy))

    strongest = np.partition(
        energy,
        -K
    )[-K:]

    return np.sum(strongest) / total


def mutual_coherence(A):
    """
    Max normalized inner product between distinct dictionary atoms.
    """
    norms = np.linalg.norm(
        A,
        axis=0,
        keepdims=True
    )

    A_n = A / np.maximum(norms, 1e-15)

    G = np.abs(
        A_n.conj().T @ A_n
    )

    np.fill_diagonal(G, 0.0)

    return np.max(G)


def condition_number(A):
    """
    Condition number from singular values.
    """
    s = np.linalg.svd(
        A,
        compute_uv=False
    )

    return (
        s[0]
        / np.maximum(s[-1], 1e-15)
    )


# ============================================================
# 9. MAIN EXPERIMENT
# ============================================================

def run_experiment():
    rng = np.random.default_rng(SEED)

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    dft_nmse = {
        K: np.zeros((Nreal, len(b_list)))
        for K in K_list
    }

    geom_nmse = {
        K: np.zeros((Nreal, len(b_list)))
        for K in K_list
    }

    dft_concentration = {
        K: np.zeros((Nreal, len(b_list)))
        for K in K_list
    }

    geom_concentration = {
        K: np.zeros((Nreal, len(b_list)))
        for K in K_list
    }

    oracle_nmse = np.zeros(
        (Nreal, len(b_list))
    )

    mu_B = np.zeros(
        (Nreal, len(b_list))
    )

    mu_U = np.zeros(
        (Nreal, len(b_list))
    )

    cond_B = np.zeros(
        (Nreal, len(b_list))
    )

    cond_U = np.zeros(
        (Nreal, len(b_list))
    )

    print("=" * 80)
    print("FIM REPRESENTATION QUALITY DIAGNOSTICS")
    print("=" * 80)
    print(
        f"N_B={N_B}, N_U={N_U}, "
        f"L={L_paths}, M={M}, Nreal={Nreal}"
    )
    print(
        f"Dictionary atoms per array = "
        f"{N_AZ} x {N_EL} = {N_AZ*N_EL}"
    )
    print("=" * 80)

    # ========================================================
    # Realizations
    # ========================================================

    for r in range(Nreal):

        # Same physical paths for every b
        paths = generate_path_parameters(rng)

        # Same normalized shapes for every b
        profiles_B, profiles_U = (
            generate_deformation_codebook(rng)
        )

        for ib, b_norm in enumerate(b_list):

            # Average over M deformation states
            tmp_dft_nmse = {
                K: [] for K in K_list
            }

            tmp_geom_nmse = {
                K: [] for K in K_list
            }

            tmp_dft_conc = {
                K: [] for K in K_list
            }

            tmp_geom_conc = {
                K: [] for K in K_list
            }

            tmp_oracle = []

            tmp_mu_B = []
            tmp_mu_U = []

            tmp_cond_B = []
            tmp_cond_U = []

            for m in range(M):

                P_Bm = apply_deformation(
                    P_B0,
                    profiles_B[m],
                    b_norm
                )

                P_Um = apply_deformation(
                    P_U0,
                    profiles_U[m],
                    b_norm
                )

                Hm = build_channel(
                    P_Bm,
                    P_Um,
                    paths
                )

                # ====================================================
                # A) FIXED DFT
                # ====================================================

                S_dft = fixed_dft_coefficients(Hm)

                for K in K_list:
                    S_dft_K = retain_top_k(
                        S_dft,
                        K
                    )

                    Hhat_dft = reconstruct_from_dft(
                        S_dft_K
                    )

                    tmp_dft_nmse[K].append(
                        nmse(
                            Hm,
                            Hhat_dft
                        )
                    )

                    tmp_dft_conc[K].append(
                        top_k_concentration(
                            S_dft,
                            K
                        )
                    )

                # ====================================================
                # B) GEOMETRY-AWARE DICTIONARY
                # ====================================================

                S_geom, A_B, A_U = (
                    geometry_dictionary_coefficients(
                        Hm,
                        P_Bm,
                        P_Um
                    )
                )

                for K in K_list:
                    S_geom_K = retain_top_k(
                        S_geom,
                        K
                    )

                    Hhat_geom = (
                        reconstruct_from_geometry_dictionary(
                            S_geom_K,
                            A_B,
                            A_U
                        )
                    )

                    tmp_geom_nmse[K].append(
                        nmse(
                            Hm,
                            Hhat_geom
                        )
                    )

                    tmp_geom_conc[K].append(
                        top_k_concentration(
                            S_geom,
                            K
                        )
                    )

                # ====================================================
                # C) ORACLE PATH-BASIS SANITY TEST
                # ====================================================

                H_oracle = oracle_path_reconstruction(
                    P_Bm,
                    P_Um,
                    paths
                )

                tmp_oracle.append(
                    nmse(
                        Hm,
                        H_oracle
                    )
                )

                # ====================================================
                # D) DICTIONARY COHERENCE / CONDITIONING
                # ====================================================

                tmp_mu_B.append(
                    mutual_coherence(A_B)
                )

                tmp_mu_U.append(
                    mutual_coherence(A_U)
                )

                tmp_cond_B.append(
                    condition_number(A_B)
                )

                tmp_cond_U.append(
                    condition_number(A_U)
                )

            # --------------------------------------------------------
            # Average over M deformation states
            # --------------------------------------------------------

            for K in K_list:
                dft_nmse[K][r, ib] = np.mean(
                    tmp_dft_nmse[K]
                )

                geom_nmse[K][r, ib] = np.mean(
                    tmp_geom_nmse[K]
                )

                dft_concentration[K][r, ib] = np.mean(
                    tmp_dft_conc[K]
                )

                geom_concentration[K][r, ib] = np.mean(
                    tmp_geom_conc[K]
                )

            oracle_nmse[r, ib] = np.mean(
                tmp_oracle
            )

            mu_B[r, ib] = np.mean(
                tmp_mu_B
            )

            mu_U[r, ib] = np.mean(
                tmp_mu_U
            )

            cond_B[r, ib] = np.mean(
                tmp_cond_B
            )

            cond_U[r, ib] = np.mean(
                tmp_cond_U
            )

        if (
            r == 0
            or (r + 1) % 20 == 0
        ):
            print(
                f"Completed {r+1}/{Nreal}"
            )

    return {
        "dft_nmse": dft_nmse,
        "geom_nmse": geom_nmse,
        "dft_concentration": dft_concentration,
        "geom_concentration": geom_concentration,
        "oracle_nmse": oracle_nmse,
        "mu_B": mu_B,
        "mu_U": mu_U,
        "cond_B": cond_B,
        "cond_U": cond_U,
    }


# ============================================================
# 10. PRINT SUMMARY
# ============================================================

def print_summary(results, K=10):
    print("\n")
    print("=" * 120)
    print(f"SUMMARY FOR TOP-{K}")
    print("=" * 120)

    print(
        " b/lambda | "
        "DFT NMSE[dB] | "
        "Geom NMSE[dB] | "
        "DFT C_K | "
        "Geom C_K | "
        "mu_B | "
        "mu_U | "
        "cond_B | "
        "cond_U | "
        "Oracle NMSE[dB]"
    )

    print("-" * 120)

    for ib, b in enumerate(b_list):

        dft_db = to_db(
            np.mean(
                results["dft_nmse"][K][:, ib]
            )
        )

        geom_db = to_db(
            np.mean(
                results["geom_nmse"][K][:, ib]
            )
        )

        dft_c = np.mean(
            results[
                "dft_concentration"
            ][K][:, ib]
        )

        geom_c = np.mean(
            results[
                "geom_concentration"
            ][K][:, ib]
        )

        mu_b = np.mean(
            results["mu_B"][:, ib]
        )

        mu_u = np.mean(
            results["mu_U"][:, ib]
        )

        c_b = np.mean(
            results["cond_B"][:, ib]
        )

        c_u = np.mean(
            results["cond_U"][:, ib]
        )

        oracle_db = to_db(
            np.mean(
                results[
                    "oracle_nmse"
                ][:, ib]
            )
        )

        print(
            f"{b:9.2f} | "
            f"{dft_db:12.3f} | "
            f"{geom_db:13.3f} | "
            f"{dft_c:7.4f} | "
            f"{geom_c:8.4f} | "
            f"{mu_b:5.3f} | "
            f"{mu_u:5.3f} | "
            f"{c_b:7.2f} | "
            f"{c_u:7.2f} | "
            f"{oracle_db:15.3f}"
        )


# ============================================================
# 11. PLOT: TOP-K RECONSTRUCTION NMSE
# ============================================================

def plot_reconstruction_nmse(
    results,
    K=10
):
    dft = results["dft_nmse"][K]
    geom = results["geom_nmse"][K]

    dft_mean = np.mean(
        dft,
        axis=0
    )

    geom_mean = np.mean(
        geom,
        axis=0
    )

    plt.figure(
        figsize=(8, 5.5)
    )

    plt.plot(
        b_list,
        to_db(dft_mean),
        marker="o",
        linewidth=2,
        label=f"Fixed UPA DFT, Top-{K}"
    )

    plt.plot(
        b_list,
        to_db(geom_mean),
        marker="s",
        linewidth=2,
        label=f"Geometry-aware, Top-{K}"
    )

    plt.xlabel(
        r"Morphing range $b/\lambda$"
    )

    plt.ylabel(
        "Reconstruction NMSE (dB)"
    )

    plt.title(
        f"Top-{K} channel reconstruction vs. morphing range"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"reconstruction_nmse_top{K}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 12. PLOT: MULTIPLE K VALUES
# ============================================================

def plot_multiple_K_nmse(
    results,
    method="dft"
):
    plt.figure(
        figsize=(8, 5.5)
    )

    key = (
        "dft_nmse"
        if method == "dft"
        else "geom_nmse"
    )

    method_name = (
        "Fixed UPA DFT"
        if method == "dft"
        else "Geometry-aware dictionary"
    )

    for K in K_list:
        mean_nmse = np.mean(
            results[key][K],
            axis=0
        )

        plt.plot(
            b_list,
            to_db(mean_nmse),
            marker="o",
            linewidth=2,
            label=f"K={K}"
        )

    plt.xlabel(
        r"Morphing range $b/\lambda$"
    )

    plt.ylabel(
        "Reconstruction NMSE (dB)"
    )

    plt.title(
        f"{method_name}: sparse reconstruction quality"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"{method}_nmse_multiple_K.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 13. PLOT: MUTUAL COHERENCE
# ============================================================

def plot_mutual_coherence(results):

    mu_B_mean = np.mean(
        results["mu_B"],
        axis=0
    )

    mu_U_mean = np.mean(
        results["mu_U"],
        axis=0
    )

    plt.figure(
        figsize=(8, 5.5)
    )

    plt.plot(
        b_list,
        mu_B_mean,
        marker="o",
        linewidth=2,
        label="BS dictionary"
    )

    plt.plot(
        b_list,
        mu_U_mean,
        marker="s",
        linewidth=2,
        label="UE dictionary"
    )

    plt.xlabel(
        r"Morphing range $b/\lambda$"
    )

    plt.ylabel(
        "Mutual coherence"
    )

    plt.title(
        "Geometry-aware dictionary coherence vs. morphing range"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "dictionary_mutual_coherence.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 14. PLOT: CONDITION NUMBER
# ============================================================

def plot_condition_number(results):

    cond_B_mean = np.mean(
        results["cond_B"],
        axis=0
    )

    cond_U_mean = np.mean(
        results["cond_U"],
        axis=0
    )

    plt.figure(
        figsize=(8, 5.5)
    )

    plt.semilogy(
        b_list,
        cond_B_mean,
        marker="o",
        linewidth=2,
        label="BS dictionary"
    )

    plt.semilogy(
        b_list,
        cond_U_mean,
        marker="s",
        linewidth=2,
        label="UE dictionary"
    )

    plt.xlabel(
        r"Morphing range $b/\lambda$"
    )

    plt.ylabel(
        "Condition number"
    )

    plt.title(
        "Geometry-aware dictionary conditioning vs. morphing range"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "dictionary_condition_number.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 15. PLOT: ORACLE SANITY TEST
# ============================================================

def plot_oracle_nmse(results):

    oracle_mean = np.mean(
        results[
            "oracle_nmse"
        ],
        axis=0
    )

    plt.figure(
        figsize=(8, 5.5)
    )

    plt.plot(
        b_list,
        to_db(oracle_mean),
        marker="o",
        linewidth=2
    )

    plt.xlabel(
        r"Morphing range $b/\lambda$"
    )

    plt.ylabel(
        "Oracle reconstruction NMSE (dB)"
    )

    plt.title(
        "Oracle geometry/path-basis sanity test"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "oracle_reconstruction_nmse.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 16. OPTIONAL: CORRELATION BETWEEN REPRESENTATION QUALITY
#     AND MORPHING RANGE
# ============================================================

def print_trend_correlations(
    results,
    K=10
):
    """
    Correlation between b and each mean diagnostic.

    This is not the CNN-NMSE correlation yet.
    It simply summarizes monotonic trends in this diagnostic test.
    """

    dft_nmse_mean = np.mean(
        results["dft_nmse"][K],
        axis=0
    )

    geom_nmse_mean = np.mean(
        results["geom_nmse"][K],
        axis=0
    )

    mu_B_mean = np.mean(
        results["mu_B"],
        axis=0
    )

    cond_B_mean = np.mean(
        results["cond_B"],
        axis=0
    )

    print("\nTrend correlations with b/lambda:")
    print(
        f"corr(b, DFT NMSE)  = "
        f"{np.corrcoef(b_list, dft_nmse_mean)[0,1]:.4f}"
    )

    print(
        f"corr(b, Geom NMSE) = "
        f"{np.corrcoef(b_list, geom_nmse_mean)[0,1]:.4f}"
    )

    print(
        f"corr(b, mu_B)      = "
        f"{np.corrcoef(b_list, mu_B_mean)[0,1]:.4f}"
    )

    print(
        f"corr(b, cond_B)    = "
        f"{np.corrcoef(b_list, cond_B_mean)[0,1]:.4f}"
    )


# ============================================================
# 17. SAVE RESULTS
# ============================================================

def save_results(results):

    save_dict = {
        "b_list": b_list,
        "oracle_nmse": results[
            "oracle_nmse"
        ],
        "mu_B": results[
            "mu_B"
        ],
        "mu_U": results[
            "mu_U"
        ],
        "cond_B": results[
            "cond_B"
        ],
        "cond_U": results[
            "cond_U"
        ],
    }

    for K in K_list:
        save_dict[
            f"dft_nmse_K{K}"
        ] = results[
            "dft_nmse"
        ][K]

        save_dict[
            f"geom_nmse_K{K}"
        ] = results[
            "geom_nmse"
        ][K]

        save_dict[
            f"dft_concentration_K{K}"
        ] = results[
            "dft_concentration"
        ][K]

        save_dict[
            f"geom_concentration_K{K}"
        ] = results[
            "geom_concentration"
        ][K]

    np.savez(
        OUTPUT_DIR
        / "representation_quality_results.npz",
        **save_dict
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    results = run_experiment()

    # Primary numerical summary
    print_summary(
        results,
        K=10
    )

    print_trend_correlations(
        results,
        K=10
    )

    save_results(
        results
    )

    # Most important figure
    plot_reconstruction_nmse(
        results,
        K=10
    )

    # Supporting diagnostics
    plot_multiple_K_nmse(
        results,
        method="dft"
    )

    plot_multiple_K_nmse(
        results,
        method="geometry"
    )

    plot_mutual_coherence(
        results
    )

    plot_condition_number(
        results
    )

    plot_oracle_nmse(
        results
    )
