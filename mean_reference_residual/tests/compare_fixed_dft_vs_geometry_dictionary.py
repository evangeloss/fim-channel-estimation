"""
compare_fixed_dft_vs_geometry_dictionary.py

Goal
----
Compare how beamspace/channel representation concentration changes with
morphing range b/lambda for:

1) Fixed UPA DFT beamspace
2) Geometry-aware steering dictionary

The important hypothesis is:

    C_q^DFT(b) decreases with b

while ideally

    C_q^GEOM(b) remains flatter.

The geometry-aware dictionary is reconstructed from the ACTUAL deformed
antenna coordinates for every deformation state.

IMPORTANT:
Within each Monte-Carlo realization:
    - channel path parameters are fixed across all b values
    - normalized deformation shapes are fixed across all b values
    - only deformation amplitude changes

This isolates the effect of morphing range.
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

# ------------------------------------------------------------
# Array dimensions
# ------------------------------------------------------------

NH_B = 5
NV_B = 5

NH_U = 5
NV_U = 5

N_B = NH_B * NV_B
N_U = NH_U * NV_U

dx_B = wavelength / 8
dy_B = wavelength / 8

dx_U = wavelength / 8
dy_U = wavelength / 8


# ------------------------------------------------------------
# Channel parameters
# ------------------------------------------------------------

L_paths = 3

# Number of deformation observations
M = 8

# Monte-Carlo channel realizations
Nreal = 200


# ------------------------------------------------------------
# Morphing range
# ------------------------------------------------------------

b_list = np.array([
    0.00,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50
])


# ------------------------------------------------------------
# Concentration metrics
# ------------------------------------------------------------

top_q_list = [
    1,
    5,
    10,
    25
]


# ------------------------------------------------------------
# Steering dictionary angular grid
#
# 5 x 5 = 25 steering vectors
#
# This deliberately gives the geometry dictionary 25 basis
# vectors per array, matching the 25-dimensional DFT basis.
#
# This makes Top-1, Top-5, Top-10, etc. directly interpretable.
# ------------------------------------------------------------

N_AZ = 5
N_EL = 5

AZ_MIN = -np.pi / 2
AZ_MAX = +np.pi / 2

EL_MIN = -np.pi / 3
EL_MAX = +np.pi / 3


# Regularization for dictionary inversion
REGULARIZATION = 1e-4


OUTPUT_DIR = Path(
    "dft_vs_geometry_dictionary"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. ARRAY GEOMETRY
# ============================================================

def create_upa_positions(
    NH,
    NV,
    dx,
    dy
):

    x = (
        np.arange(NH)
        - (NH - 1) / 2
    ) * dx

    y = (
        np.arange(NV)
        - (NV - 1) / 2
    ) * dy

    X, Y = np.meshgrid(
        x,
        y,
        indexing="ij"
    )

    positions = np.column_stack([
        X.reshape(-1),
        Y.reshape(-1),
        np.zeros(NH * NV)
    ])

    return positions


P_B0 = create_upa_positions(
    NH_B,
    NV_B,
    dx_B,
    dy_B
)

P_U0 = create_upa_positions(
    NH_U,
    NV_U,
    dx_U,
    dy_U
)


# ============================================================
# 3. DFT BEAMSPACE
# ============================================================

def unitary_dft(N):

    n = np.arange(N)
    k = n[:, None]

    F = (
        np.exp(
            -1j
            * 2
            * np.pi
            * k
            * n
            / N
        )
        / np.sqrt(N)
    )

    return F


def upa_dft(
    NH,
    NV
):

    F_H = unitary_dft(NH)
    F_V = unitary_dft(NV)

    # Matches the flattening convention used above
    return np.kron(
        F_H,
        F_V
    )


F_B = upa_dft(
    NH_B,
    NV_B
)

F_U = upa_dft(
    NH_U,
    NV_U
)


def fixed_dft_transform(H):

    return (
        F_B.conj().T
        @ H
        @ F_U
    )


# ============================================================
# 4. PHYSICAL STEERING MODEL
# ============================================================

def direction_vector(
    azimuth,
    elevation
):

    return np.array([
        np.cos(elevation)
        * np.cos(azimuth),

        np.cos(elevation)
        * np.sin(azimuth),

        np.sin(elevation)
    ])


def steering_vector(
    positions,
    direction
):

    phase = (
        2
        * np.pi
        / wavelength
        * positions
        @ direction
    )

    a = np.exp(
        1j * phase
    )

    return (
        a / np.sqrt(
            len(a)
        )
    )


# ============================================================
# 5. CHANNEL GENERATION
# ============================================================

def generate_path_parameters(
    rng
):

    paths = []

    for _ in range(L_paths):

        az_B = rng.uniform(
            AZ_MIN,
            AZ_MAX
        )

        el_B = rng.uniform(
            EL_MIN,
            EL_MAX
        )

        az_U = rng.uniform(
            AZ_MIN,
            AZ_MAX
        )

        el_U = rng.uniform(
            EL_MIN,
            EL_MAX
        )

        u_B = direction_vector(
            az_B,
            el_B
        )

        u_U = direction_vector(
            az_U,
            el_U
        )

        alpha = (
            rng.normal()
            + 1j * rng.normal()
        ) / np.sqrt(
            2 * L_paths
        )

        paths.append({
            "alpha": alpha,
            "u_B": u_B,
            "u_U": u_U
        })

    return paths


def build_channel(
    P_B,
    P_U,
    paths
):

    H = np.zeros(
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

        H += (
            path["alpha"]
            * np.outer(
                a_B,
                np.conj(a_U)
            )
        )

    return H


# ============================================================
# 6. DEFORMATION MODEL
# ============================================================

def normalize_profile(z):

    z = (
        z
        - np.mean(z)
    )

    max_value = np.max(
        np.abs(z)
    )

    if max_value > 1e-12:

        z = (
            z
            / max_value
        )

    return z


def generate_single_profile(
    NH,
    NV,
    rng
):

    x = np.linspace(
        -1,
        1,
        NH
    )

    y = np.linspace(
        -1,
        1,
        NV
    )

    X, Y = np.meshgrid(
        x,
        y,
        indexing="ij"
    )

    deformation_type = (
        rng.integers(
            0,
            4
        )
    )


    # --------------------------------------------------------
    # Parabolic
    # --------------------------------------------------------

    if deformation_type == 0:

        profile = (
            X**2
            + 0.7 * Y**2
        )


    # --------------------------------------------------------
    # Sinusoidal
    # --------------------------------------------------------

    elif deformation_type == 1:

        phase = rng.uniform(
            0,
            2 * np.pi
        )

        profile = (
            np.sin(
                np.pi * X
                + phase
            )
            * np.cos(
                np.pi * Y
            )
        )


    # --------------------------------------------------------
    # Tilted
    # --------------------------------------------------------

    elif deformation_type == 2:

        ax = rng.uniform(
            -1,
            1
        )

        ay = rng.uniform(
            -1,
            1
        )

        profile = (
            ax * X
            + ay * Y
        )


    # --------------------------------------------------------
    # Mixed
    # --------------------------------------------------------

    else:

        a = rng.uniform(
            -1,
            1
        )

        b = rng.uniform(
            -1,
            1
        )

        d = rng.uniform(
            -1,
            1
        )

        profile = (
            a * X
            + b * Y
            + d
            * np.sin(
                np.pi * X
            )
            * np.cos(
                np.pi * Y
            )
        )


    return normalize_profile(
        profile.reshape(-1)
    )


def generate_deformation_codebook(
    rng
):

    profiles_B = []
    profiles_U = []

    for _ in range(M):

        profiles_B.append(
            generate_single_profile(
                NH_B,
                NV_B,
                rng
            )
        )

        profiles_U.append(
            generate_single_profile(
                NH_U,
                NV_U,
                rng
            )
        )

    return (
        np.stack(
            profiles_B
        ),
        np.stack(
            profiles_U
        )
    )


def apply_deformation(
    P0,
    profile,
    b_over_lambda
):

    P = P0.copy()

    dz = (
        b_over_lambda
        * wavelength
        * profile
    )

    P[:, 2] += dz

    return P


# ============================================================
# 7. GEOMETRY-AWARE STEERING DICTIONARY
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


def build_geometry_dictionary(
    positions
):
    """
    Build steering dictionary using the ACTUAL array positions.

    If the surface bends, the dictionary therefore bends with it.

    Output:
        A : [N_antennas, N_AZ*N_EL]
    """

    columns = []

    for azimuth in azimuth_grid:

        for elevation in elevation_grid:

            u = direction_vector(
                azimuth,
                elevation
            )

            a = steering_vector(
                positions,
                u
            )

            columns.append(
                a
            )

    A = np.column_stack(
        columns
    )

    return A


# ============================================================
# 8. REGULARIZED LEFT INVERSE
# ============================================================

def regularized_left_inverse(
    A,
    reg=REGULARIZATION
):
    """
    L = (A^H A + lambda I)^(-1) A^H

    This is more numerically stable than a direct inverse.
    """

    G = (
        A.conj().T
        @ A
    )

    I = np.eye(
        G.shape[0],
        dtype=G.dtype
    )

    L = np.linalg.solve(
        G + reg * I,
        A.conj().T
    )

    return L


# ============================================================
# 9. GEOMETRY-AWARE CHANNEL REPRESENTATION
# ============================================================

def geometry_dictionary_transform(
    H,
    P_B,
    P_U
):
    """
    Approximate:

        H = A_B S A_U^H

    Therefore

        S ≈ A_B^dagger H (A_U^dagger)^H

    using regularized inverses.
    """

    A_B = build_geometry_dictionary(
        P_B
    )

    A_U = build_geometry_dictionary(
        P_U
    )

    L_B = regularized_left_inverse(
        A_B
    )

    L_U = regularized_left_inverse(
        A_U
    )

    S = (
        L_B
        @ H
        @ L_U.conj().T
    )

    return S


# ============================================================
# 10. CONCENTRATION METRICS
# ============================================================

def top_q_energy_concentration(
    X,
    q
):

    energy = (
        np.abs(
            X.reshape(-1)
        )**2
    )

    total = np.sum(
        energy
    )

    if total < 1e-15:
        return 0.0

    q = min(
        q,
        len(energy)
    )

    strongest = np.partition(
        energy,
        -q
    )[-q:]

    return (
        np.sum(strongest)
        / total
    )


def normalized_entropy(X):

    energy = (
        np.abs(
            X.reshape(-1)
        )**2
    )

    p = (
        energy
        / (
            np.sum(energy)
            + 1e-15
        )
    )

    p = p[
        p > 1e-15
    ]

    entropy = -np.sum(
        p * np.log(p)
    )

    entropy /= np.log(
        X.size
    )

    return entropy


def support_for_energy(
    X,
    target=0.90
):

    energy = (
        np.abs(
            X.reshape(-1)
        )**2
    )

    energy = np.sort(
        energy
    )[::-1]

    cumulative = np.cumsum(
        energy
    )

    cumulative /= (
        cumulative[-1]
        + 1e-15
    )

    return (
        np.searchsorted(
            cumulative,
            target
        )
        + 1
    )


# ============================================================
# 11. MAIN MONTE-CARLO EXPERIMENT
# ============================================================

def run_experiment():

    rng = np.random.default_rng(
        SEED
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    dft_concentration = {
        q: np.zeros(
            (
                Nreal,
                len(b_list)
            )
        )
        for q in top_q_list
    }

    geom_concentration = {
        q: np.zeros(
            (
                Nreal,
                len(b_list)
            )
        )
        for q in top_q_list
    }


    dft_entropy = np.zeros(
        (
            Nreal,
            len(b_list)
        )
    )

    geom_entropy = np.zeros(
        (
            Nreal,
            len(b_list)
        )
    )


    dft_k90 = np.zeros(
        (
            Nreal,
            len(b_list)
        )
    )

    geom_k90 = np.zeros(
        (
            Nreal,
            len(b_list)
        )
    )


    print("=" * 70)

    print(
        "Fixed DFT vs Geometry-Aware Dictionary"
    )

    print("=" * 70)

    print(
        f"N_B = {N_B}, "
        f"N_U = {N_U}"
    )

    print(
        f"L = {L_paths}, "
        f"M = {M}, "
        f"Nreal = {Nreal}"
    )

    print(
        f"Geometry dictionary size = "
        f"{N_AZ} x {N_EL} "
        f"= {N_AZ*N_EL}"
    )

    print("=" * 70)


    # ========================================================
    # Monte-Carlo realizations
    # ========================================================

    for realization in range(
        Nreal
    ):

        # ----------------------------------------------------
        # SAME channel paths for every b
        # ----------------------------------------------------

        paths = (
            generate_path_parameters(
                rng
            )
        )


        # ----------------------------------------------------
        # SAME normalized deformation shapes for every b
        # ----------------------------------------------------

        profiles_B, profiles_U = (
            generate_deformation_codebook(
                rng
            )
        )


        # ====================================================
        # Sweep b
        # ====================================================

        for ib, b_norm in enumerate(
            b_list
        ):

            temp_dft = {
                q: []
                for q in top_q_list
            }

            temp_geom = {
                q: []
                for q in top_q_list
            }

            temp_dft_entropy = []
            temp_geom_entropy = []

            temp_dft_k90 = []
            temp_geom_k90 = []


            # ================================================
            # M deformation observations
            # ================================================

            for m in range(M):

                # --------------------------------------------
                # Actual geometry at deformation m
                # --------------------------------------------

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


                # --------------------------------------------
                # SAME physical channel paths
                # --------------------------------------------

                Hm = build_channel(
                    P_Bm,
                    P_Um,
                    paths
                )


                # ============================================
                # Representation 1:
                # FIXED flat-array DFT
                # ============================================

                H_dft = (
                    fixed_dft_transform(
                        Hm
                    )
                )


                # ============================================
                # Representation 2:
                # GEOMETRY-AWARE dictionary
                # ============================================

                H_geom = (
                    geometry_dictionary_transform(
                        Hm,
                        P_Bm,
                        P_Um
                    )
                )


                # ============================================
                # Top-q concentration
                # ============================================

                for q in top_q_list:

                    temp_dft[q].append(
                        top_q_energy_concentration(
                            H_dft,
                            q
                        )
                    )

                    temp_geom[q].append(
                        top_q_energy_concentration(
                            H_geom,
                            q
                        )
                    )


                # ============================================
                # Entropy
                # ============================================

                temp_dft_entropy.append(
                    normalized_entropy(
                        H_dft
                    )
                )

                temp_geom_entropy.append(
                    normalized_entropy(
                        H_geom
                    )
                )


                # ============================================
                # K90
                # ============================================

                temp_dft_k90.append(
                    support_for_energy(
                        H_dft,
                        target=0.90
                    )
                )

                temp_geom_k90.append(
                    support_for_energy(
                        H_geom,
                        target=0.90
                    )
                )


            # ================================================
            # Average across the M deformation observations
            # ================================================

            for q in top_q_list:

                dft_concentration[q][
                    realization,
                    ib
                ] = np.mean(
                    temp_dft[q]
                )

                geom_concentration[q][
                    realization,
                    ib
                ] = np.mean(
                    temp_geom[q]
                )


            dft_entropy[
                realization,
                ib
            ] = np.mean(
                temp_dft_entropy
            )

            geom_entropy[
                realization,
                ib
            ] = np.mean(
                temp_geom_entropy
            )


            dft_k90[
                realization,
                ib
            ] = np.mean(
                temp_dft_k90
            )

            geom_k90[
                realization,
                ib
            ] = np.mean(
                temp_geom_k90
            )


        if (
            realization == 0
            or (realization + 1) % 20 == 0
        ):

            print(
                f"Completed "
                f"{realization + 1}"
                f"/{Nreal}"
            )


    return {
        "dft_concentration":
            dft_concentration,

        "geom_concentration":
            geom_concentration,

        "dft_entropy":
            dft_entropy,

        "geom_entropy":
            geom_entropy,

        "dft_k90":
            dft_k90,

        "geom_k90":
            geom_k90
    }


# ============================================================
# 12. PRINT RESULTS
# ============================================================

def print_results(results):

    print("\n")
    print("=" * 95)

    print(
        "MEAN CONCENTRATION RESULTS"
    )

    print("=" * 95)

    print(
        " b/lambda | "
        "DFT C10 | "
        "Geometry C10 | "
        "DFT C25 | "
        "Geometry C25"
    )

    print("-" * 95)


    for ib, b in enumerate(
        b_list
    ):

        dft10 = np.mean(
            results[
                "dft_concentration"
            ][10][:, ib]
        )

        geo10 = np.mean(
            results[
                "geom_concentration"
            ][10][:, ib]
        )

        dft25 = np.mean(
            results[
                "dft_concentration"
            ][25][:, ib]
        )

        geo25 = np.mean(
            results[
                "geom_concentration"
            ][25][:, ib]
        )


        print(
            f"{b:9.2f} | "
            f"{dft10:7.4f} | "
            f"{geo10:12.4f} | "
            f"{dft25:7.4f} | "
            f"{geo25:12.4f}"
        )


# ============================================================
# 13. PLOT TOP-q COMPARISON
# ============================================================

def plot_topq_comparison(
    results,
    q=10
):

    dft = (
        results[
            "dft_concentration"
        ][q]
    )

    geom = (
        results[
            "geom_concentration"
        ][q]
    )


    dft_mean = np.mean(
        dft,
        axis=0
    )

    geom_mean = np.mean(
        geom,
        axis=0
    )


    dft_sem = (
        np.std(
            dft,
            axis=0
        )
        / np.sqrt(Nreal)
    )

    geom_sem = (
        np.std(
            geom,
            axis=0
        )
        / np.sqrt(Nreal)
    )


    plt.figure(
        figsize=(8, 5.5)
    )


    plt.plot(
        b_list,
        dft_mean,
        marker="o",
        linewidth=2,
        label="Fixed UPA DFT"
    )

    plt.fill_between(
        b_list,
        dft_mean - 1.96*dft_sem,
        dft_mean + 1.96*dft_sem,
        alpha=0.15
    )


    plt.plot(
        b_list,
        geom_mean,
        marker="s",
        linewidth=2,
        label="Geometry-aware dictionary"
    )

    plt.fill_between(
        b_list,
        geom_mean - 1.96*geom_sem,
        geom_mean + 1.96*geom_sem,
        alpha=0.15
    )


    plt.xlabel(
        r"Morphing range $b/\lambda$"
    )

    plt.ylabel(
        rf"Top-{q} energy concentration $C_{{{q}}}$"
    )

    plt.title(
        rf"Fixed DFT vs. geometry-aware representation ($C_{{{q}}}$)"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()


    plt.savefig(
        OUTPUT_DIR
        / f"C{q}_DFT_vs_geometry.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 14. NORMALIZED CONCENTRATION
#
# This is particularly important for the paper:
#
#       C_q(b) / C_q(0)
#
# It compares how much concentration each representation LOSES
# relative to its own undeformed baseline.
# ============================================================

def plot_relative_concentration(
    results,
    q=10
):

    dft_mean = np.mean(
        results[
            "dft_concentration"
        ][q],
        axis=0
    )

    geom_mean = np.mean(
        results[
            "geom_concentration"
        ][q],
        axis=0
    )


    dft_relative = (
        dft_mean
        / dft_mean[0]
    )

    geom_relative = (
        geom_mean
        / geom_mean[0]
    )


    plt.figure(
        figsize=(8, 5.5)
    )


    plt.plot(
        b_list,
        dft_relative,
        marker="o",
        linewidth=2,
        label="Fixed UPA DFT"
    )

    plt.plot(
        b_list,
        geom_relative,
        marker="s",
        linewidth=2,
        label="Geometry-aware dictionary"
    )


    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1
    )


    plt.xlabel(
        r"Morphing range $b/\lambda$"
    )

    plt.ylabel(
        rf"Relative concentration $C_{{{q}}}(b)/C_{{{q}}}(0)$"
    )

    plt.title(
        "Representation robustness to FIM deformation"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()


    plt.savefig(
        OUTPUT_DIR
        / f"C{q}_relative_DFT_vs_geometry.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 15. ENTROPY COMPARISON
# ============================================================

def plot_entropy(
    results
):

    dft = np.mean(
        results[
            "dft_entropy"
        ],
        axis=0
    )

    geom = np.mean(
        results[
            "geom_entropy"
        ],
        axis=0
    )


    plt.figure(
        figsize=(8, 5.5)
    )


    plt.plot(
        b_list,
        dft,
        marker="o",
        linewidth=2,
        label="Fixed UPA DFT"
    )

    plt.plot(
        b_list,
        geom,
        marker="s",
        linewidth=2,
        label="Geometry-aware dictionary"
    )


    plt.xlabel(
        r"Morphing range $b/\lambda$"
    )

    plt.ylabel(
        "Normalized coefficient entropy"
    )

    plt.title(
        "Representation spreading vs. morphing range"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()


    plt.savefig(
        OUTPUT_DIR
        / "entropy_DFT_vs_geometry.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 16. K90 COMPARISON
# ============================================================

def plot_k90(
    results
):

    dft = np.mean(
        results[
            "dft_k90"
        ],
        axis=0
    )

    geom = np.mean(
        results[
            "geom_k90"
        ],
        axis=0
    )


    plt.figure(
        figsize=(8, 5.5)
    )


    plt.plot(
        b_list,
        dft,
        marker="o",
        linewidth=2,
        label="Fixed UPA DFT"
    )

    plt.plot(
        b_list,
        geom,
        marker="s",
        linewidth=2,
        label="Geometry-aware dictionary"
    )


    plt.xlabel(
        r"Morphing range $b/\lambda$"
    )

    plt.ylabel(
        r"$K_{90}$"
    )

    plt.title(
        "Number of coefficients required for 90% energy"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()


    plt.savefig(
        OUTPUT_DIR
        / "K90_DFT_vs_geometry.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 17. SAVE RESULTS
# ============================================================

def save_results(
    results
):

    save_dict = {
        "b_list":
            b_list,

        "dft_entropy":
            results[
                "dft_entropy"
            ],

        "geom_entropy":
            results[
                "geom_entropy"
            ],

        "dft_k90":
            results[
                "dft_k90"
            ],

        "geom_k90":
            results[
                "geom_k90"
            ]
    }


    for q in top_q_list:

        save_dict[
            f"dft_C{q}"
        ] = (
            results[
                "dft_concentration"
            ][q]
        )

        save_dict[
            f"geom_C{q}"
        ] = (
            results[
                "geom_concentration"
            ][q]
        )


    np.savez(
        OUTPUT_DIR
        / "dft_vs_geometry_results.npz",
        **save_dict
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    results = run_experiment()

    print_results(
        results
    )

    save_results(
        results
    )


    # --------------------------------------------------------
    # Most important plots
    # --------------------------------------------------------

    plot_topq_comparison(
        results,
        q=10
    )

    plot_relative_concentration(
        results,
        q=10
    )


    # --------------------------------------------------------
    # Supporting plots
    # --------------------------------------------------------

    plot_topq_comparison(
        results,
        q=25
    )

    plot_entropy(
        results
    )

    plot_k90(
        results
    )