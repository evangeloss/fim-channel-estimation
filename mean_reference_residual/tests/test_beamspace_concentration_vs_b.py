"""
test_beamspace_concentration_vs_b.py

Purpose
-------
Check whether the fixed DFT beamspace representation becomes less sparse /
less concentrated as the FIM morphing range b/lambda increases.

Main metric:
    C_q = energy contained in the q strongest beamspace coefficients
          ------------------------------------------------------------
                     total beamspace energy

If C_q decreases strongly with b/lambda, the fixed beamspace basis is
becoming progressively mismatched to the deformed array.

This script is intentionally independent of CNN training.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

fc = 28e9
c = 3e8
wavelength = c / fc

# Same baseline geometry we have been using
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

L_paths = 3
M = 8

# Monte-Carlo realizations
Nreal = 200

# Include b=0 so we know the undeformed reference concentration
b_list = np.array([
    0.00,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50
])

# Number of strongest beamspace coefficients whose energy we measure
top_q_list = [1, 5, 10, 25]

OUTPUT_DIR = Path("beamspace_concentration_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. ARRAY GEOMETRY
# ============================================================

def create_upa_positions(NH, NV, dx, dy):
    """
    Create NH x NV planar array centered around zero.

    Output:
        positions: [N, 3]
    """

    x = (np.arange(NH) - (NH - 1) / 2) * dx
    y = (np.arange(NV) - (NV - 1) / 2) * dy

    xx, yy = np.meshgrid(x, y, indexing="ij")

    positions = np.column_stack([
        xx.reshape(-1),
        yy.reshape(-1),
        np.zeros(NH * NV)
    ])

    return positions


P_B0 = create_upa_positions(
    NH_B, NV_B,
    dx_B, dy_B
)

P_U0 = create_upa_positions(
    NH_U, NV_U,
    dx_U, dy_U
)


# ============================================================
# 3. DFT / BEAMSPACE MATRICES
# ============================================================

def unitary_dft(N):
    """
    Unitary DFT matrix.

    F^H F = I
    """

    n = np.arange(N)
    k = n[:, None]

    F = np.exp(
        -1j * 2 * np.pi * k * n / N
    ) / np.sqrt(N)

    return F


F_B_H = unitary_dft(NH_B)
F_B_V = unitary_dft(NV_B)

F_U_H = unitary_dft(NH_U)
F_U_V = unitary_dft(NV_U)

F_B = np.kron(F_B_H, F_B_V)
F_U = np.kron(F_U_H, F_U_V)

def to_beamspace(H):
    """
    Same convention we discussed:

        H_beam = F_B^H H F_U

    H shape:
        [N_B, N_U]
    """

    return F_B.conj().T @ H @ F_U


# ============================================================
# 4. BASIC CHECK OF BEAMSPACE TRANSFORM
# ============================================================

def check_dft_unitarity():

    err_B = np.linalg.norm(
        F_B.conj().T @ F_B - np.eye(N_B)
    )

    err_U = np.linalg.norm(
        F_U.conj().T @ F_U - np.eye(N_U)
    )

    print(f"BS DFT unitarity error: {err_B:.3e}")
    print(f"UE DFT unitarity error: {err_U:.3e}")

    assert err_B < 1e-10
    assert err_U < 1e-10


# ============================================================
# 5. CHANNEL MODEL
# ============================================================

def direction_vector(azimuth, elevation):
    """
    Unit propagation vector.
    """

    return np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation)
    ])


def steering_vector(positions, direction):
    """
    Geometry-aware steering vector

        a_n = exp(j 2pi/lambda p_n^T u)

    positions: [N, 3]
    """

    phase = (
        2 * np.pi / wavelength
        * positions @ direction
    )

    a = np.exp(1j * phase)

    return a / np.sqrt(len(a))


def generate_path_parameters(rng):
    """
    Generate ONE physical multipath realization.

    These path parameters remain unchanged for all b values
    inside the same Monte-Carlo realization.
    """

    paths = []

    for _ in range(L_paths):

        # AoD
        az_B = rng.uniform(-np.pi / 2, np.pi / 2)
        el_B = rng.uniform(-np.pi / 3, np.pi / 3)

        # AoA
        az_U = rng.uniform(-np.pi / 2, np.pi / 2)
        el_U = rng.uniform(-np.pi / 3, np.pi / 3)

        u_B = direction_vector(az_B, el_B)
        u_U = direction_vector(az_U, el_U)

        # Complex path gain
        alpha = (
            rng.normal() + 1j * rng.normal()
        ) / np.sqrt(2 * L_paths)

        paths.append({
            "alpha": alpha,
            "u_B": u_B,
            "u_U": u_U
        })

    return paths


def build_channel(P_B, P_U, paths):
    """
    Geometry-dependent MIMO channel:

        H = sum_l alpha_l a_B,l a_U,l^H

    Output shape:
        [N_B, N_U]
    """

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
# 6. DEFORMATION GENERATION
# ============================================================

def normalize_profile(z):
    """
    Normalize deformation shape so

        max |z| = 1

    b/lambda will later control only its amplitude.
    """

    z = z - np.mean(z)

    maximum = np.max(np.abs(z))

    if maximum > 1e-12:
        z = z / maximum

    return z


def generate_single_profile(NH, NV, rng):
    """
    Generate one smooth mixed deformation pattern.

    This is only the normalized SHAPE.
    Actual amplitude is introduced later:

        dz = b * lambda * profile
    """

    x = np.linspace(-1, 1, NH)
    y = np.linspace(-1, 1, NV)

    X, Y = np.meshgrid(
        x,
        y,
        indexing="ij"
    )

    deformation_type = rng.integers(0, 4)

    # --------------------------------------------------------
    # Parabolic
    # --------------------------------------------------------

    if deformation_type == 0:

        profile = (
            X**2 + 0.7 * Y**2
        )

    # --------------------------------------------------------
    # Sinusoidal
    # --------------------------------------------------------

    elif deformation_type == 1:

        phase = rng.uniform(0, 2 * np.pi)

        profile = (
            np.sin(np.pi * X + phase)
            * np.cos(np.pi * Y)
        )

    # --------------------------------------------------------
    # Tilted
    # --------------------------------------------------------

    elif deformation_type == 2:

        ax = rng.uniform(-1, 1)
        ay = rng.uniform(-1, 1)

        profile = (
            ax * X + ay * Y
        )

    # --------------------------------------------------------
    # Mixed smooth deformation
    # --------------------------------------------------------

    else:

        a = rng.uniform(-1, 1)
        b = rng.uniform(-1, 1)
        d = rng.uniform(-1, 1)

        profile = (
            a * X
            + b * Y
            + d * np.sin(np.pi * X) * np.cos(np.pi * Y)
        )

    return normalize_profile(
        profile.reshape(-1)
    )


def generate_deformation_codebook(rng):
    """
    Generate M normalized deformation states.

    IMPORTANT:
    These same shapes are reused for every b/lambda value.
    """

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
        np.stack(profiles_B),
        np.stack(profiles_U)
    )


def apply_deformation(
    P0,
    normalized_profile,
    b_over_lambda
):
    """
    Deformation along z:

        dz_n = (b/lambda) * lambda * profile_n

    therefore maximum displacement is approximately

        b = (b/lambda) lambda.
    """

    P = P0.copy()

    dz = (
        b_over_lambda
        * wavelength
        * normalized_profile
    )

    P[:, 2] += dz

    return P


# ============================================================
# 7. CONCENTRATION METRICS
# ============================================================

def top_q_energy_concentration(
    H_beam,
    q
):
    """
    Fraction of total beamspace energy contained
    in the q strongest beam coefficients.

        C_q =
          sum(top-q |H_beam|^2)
          ---------------------
          sum(all |H_beam|^2)

    C_q close to 1:
        very concentrated / sparse beamspace

    smaller C_q:
        energy spread across more beams
    """

    energy = np.abs(H_beam.reshape(-1))**2

    total_energy = np.sum(energy)

    if total_energy < 1e-15:
        return 0.0

    q = min(q, len(energy))

    strongest = np.partition(
        energy,
        -q
    )[-q:]

    return (
        np.sum(strongest)
        / total_energy
    )


def normalized_entropy(H_beam):
    """
    Optional beamspace spreading metric.

    0 -> energy concentrated in one coefficient
    1 -> perfectly uniform energy

    Useful as a second sanity metric.
    """

    energy = np.abs(H_beam.reshape(-1))**2

    p = energy / (
        np.sum(energy) + 1e-15
    )

    p = p[p > 1e-15]

    entropy = -np.sum(
        p * np.log(p)
    )

    entropy /= np.log(
        H_beam.size
    )

    return entropy


def support_for_energy(
    H_beam,
    target=0.90
):
    """
    Number of strongest coefficients required
    to capture target fraction of energy.

    Smaller is better.

    Example:
        K90 = 8

    means 8 beam coefficients contain 90%
    of the channel energy.
    """

    energy = (
        np.abs(H_beam.reshape(-1))**2
    )

    energy = np.sort(energy)[::-1]

    cumulative = np.cumsum(energy)

    cumulative /= (
        cumulative[-1] + 1e-15
    )

    return (
        np.searchsorted(
            cumulative,
            target
        )
        + 1
    )


# ============================================================
# 8. MAIN EXPERIMENT
# ============================================================

def run_experiment():

    rng = np.random.default_rng(SEED)

    check_dft_unitarity()

    # results[q] -> [Nreal, len(b_list)]
    results = {
        q: np.zeros(
            (Nreal, len(b_list))
        )
        for q in top_q_list
    }

    entropy_results = np.zeros(
        (Nreal, len(b_list))
    )

    support90_results = np.zeros(
        (Nreal, len(b_list))
    )

    # Keep reference concentration too
    reference_concentration = {
        q: np.zeros(Nreal)
        for q in top_q_list
    }

    print("\nRunning beamspace concentration experiment...")
    print(
        f"N_B={N_B}, N_U={N_U}, "
        f"L={L_paths}, M={M}, Nreal={Nreal}"
    )

    for realization in range(Nreal):

        # ----------------------------------------------------
        # One physical channel realization
        # ----------------------------------------------------

        paths = generate_path_parameters(
            rng
        )

        # Reference undeformed channel
        H0 = build_channel(
            P_B0,
            P_U0,
            paths
        )

        H0_beam = to_beamspace(
            H0
        )

        for q in top_q_list:

            reference_concentration[q][
                realization
            ] = top_q_energy_concentration(
                H0_beam,
                q
            )

        # ----------------------------------------------------
        # Generate normalized deformation shapes ONCE.
        #
        # Same profiles are scaled from b=0.05 up to b=0.5.
        # ----------------------------------------------------

        profiles_B, profiles_U = (
            generate_deformation_codebook(
                rng
            )
        )

        # ----------------------------------------------------
        # Sweep morphing range
        # ----------------------------------------------------

        for ib, b_norm in enumerate(
            b_list
        ):

            # Average concentration over M observations
            concentration_M = {
                q: []
                for q in top_q_list
            }

            entropy_M = []
            support90_M = []

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

                Hm_beam = to_beamspace(
                    Hm
                )

                for q in top_q_list:

                    concentration_M[q].append(
                        top_q_energy_concentration(
                            Hm_beam,
                            q
                        )
                    )

                entropy_M.append(
                    normalized_entropy(
                        Hm_beam
                    )
                )

                support90_M.append(
                    support_for_energy(
                        Hm_beam,
                        target=0.90
                    )
                )

            # Average over M deformation observations
            for q in top_q_list:

                results[q][
                    realization,
                    ib
                ] = np.mean(
                    concentration_M[q]
                )

            entropy_results[
                realization,
                ib
            ] = np.mean(
                entropy_M
            )

            support90_results[
                realization,
                ib
            ] = np.mean(
                support90_M
            )

        if (
            (realization + 1) % 20 == 0
            or realization == 0
        ):

            print(
                f"Completed "
                f"{realization + 1}/{Nreal}"
            )

    return (
        results,
        entropy_results,
        support90_results,
        reference_concentration
    )


# ============================================================
# 9. PLOTTING
# ============================================================

def plot_concentration(results):

    plt.figure(
        figsize=(8, 5.5)
    )

    for q in top_q_list:

        mean = np.mean(
            results[q],
            axis=0
        )

        std = np.std(
            results[q],
            axis=0
        )

        stderr = (
            std / np.sqrt(Nreal)
        )

        plt.plot(
            b_list,
            mean,
            marker="o",
            linewidth=2,
            label=f"Top-{q}"
        )

        plt.fill_between(
            b_list,
            mean - 1.96 * stderr,
            mean + 1.96 * stderr,
            alpha=0.15
        )

    plt.xlabel(
        r"Morphing range $b/\lambda$"
    )

    plt.ylabel(
        r"Beamspace energy concentration $C_q$"
    )

    plt.title(
        "Beamspace concentration vs. morphing range"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "beamspace_concentration_vs_b.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


def plot_entropy(entropy_results):

    mean = np.mean(
        entropy_results,
        axis=0
    )

    stderr = (
        np.std(
            entropy_results,
            axis=0
        )
        / np.sqrt(Nreal)
    )

    plt.figure(
        figsize=(7.5, 5.2)
    )

    plt.plot(
        b_list,
        mean,
        marker="o",
        linewidth=2
    )

    plt.fill_between(
        b_list,
        mean - 1.96 * stderr,
        mean + 1.96 * stderr,
        alpha=0.15
    )

    plt.xlabel(
        r"Morphing range $b/\lambda$"
    )

    plt.ylabel(
        "Normalized beamspace entropy"
    )

    plt.title(
        "Beamspace spreading vs. morphing range"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "beamspace_entropy_vs_b.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


def plot_support90(
    support90_results
):

    mean = np.mean(
        support90_results,
        axis=0
    )

    stderr = (
        np.std(
            support90_results,
            axis=0
        )
        / np.sqrt(Nreal)
    )

    plt.figure(
        figsize=(7.5, 5.2)
    )

    plt.plot(
        b_list,
        mean,
        marker="o",
        linewidth=2
    )

    plt.fill_between(
        b_list,
        mean - 1.96 * stderr,
        mean + 1.96 * stderr,
        alpha=0.15
    )

    plt.xlabel(
        r"Morphing range $b/\lambda$"
    )

    plt.ylabel(
        r"Number of coefficients for 90% energy ($K_{90}$)"
    )

    plt.title(
        "Effective beamspace support vs. morphing range"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "beamspace_K90_vs_b.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 10. SAVE RESULTS
# ============================================================

def save_results(
    results,
    entropy_results,
    support90_results
):

    save_dict = {
        "b_list": b_list,
        "entropy": entropy_results,
        "support90": support90_results
    }

    for q in top_q_list:

        save_dict[
            f"concentration_top{q}"
        ] = results[q]

    np.savez(
        OUTPUT_DIR
        / "beamspace_concentration_results.npz",
        **save_dict
    )


# ============================================================
# 11. PRINT SUMMARY
# ============================================================

def print_summary(
    results,
    entropy_results,
    support90_results
):

    print("\n")
    print("=" * 78)
    print("BEAMSPACE CONCENTRATION RESULTS")
    print("=" * 78)

    header = (
        "b/lambda"
        + "".join(
            [
                f" | C{q:>2}"
                for q in top_q_list
            ]
        )
        + " | Entropy | K90"
    )

    print(header)
    print("-" * len(header))

    for ib, b_norm in enumerate(
        b_list
    ):

        line = f"{b_norm:8.2f}"

        for q in top_q_list:

            mean_C = np.mean(
                results[q][:, ib]
            )

            line += (
                f" | {mean_C:5.3f}"
            )

        mean_entropy = np.mean(
            entropy_results[:, ib]
        )

        mean_K90 = np.mean(
            support90_results[:, ib]
        )

        line += (
            f" | {mean_entropy:7.3f}"
            f" | {mean_K90:5.1f}"
        )

        print(line)

    print("=" * 78)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    (
        results,
        entropy_results,
        support90_results,
        reference_concentration
    ) = run_experiment()

    print_summary(
        results,
        entropy_results,
        support90_results
    )

    save_results(
        results,
        entropy_results,
        support90_results
    )

    plot_concentration(
        results
    )

    plot_entropy(
        entropy_results
    )

    plot_support90(
        support90_results
    )