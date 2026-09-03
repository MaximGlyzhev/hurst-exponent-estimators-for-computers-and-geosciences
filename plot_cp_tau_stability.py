"""Evaluate and plot the dependence of the CP estimator on ``tau``.

For each Hurst parameter H in {0.2, 0.5, 0.8}, the program generates 10,000
independent fractional Brownian motion profiles of length L = 1000. Each
profile is evaluated with the CP estimator at tau = 3, ..., 20. Importantly,
the same profiles are used at every tau, so differences between scales are
not confounded by different Monte Carlo samples.

The figure shows the signed mean estimation error, mean(H_hat - H), and a
shaded band of plus/minus one standard deviation of H_hat - H. Numerical
results are also saved as a CSV file.

References
----------
Gutjahr, T., Hale, S., Keller, K., Blum, P., & Winter, S. (2022).
Quantification of fracture roughness by change probabilities and Hurst
exponents. Mathematical Geosciences, 54, 679--710.
https://doi.org/10.1007/s11004-021-09985-3

Dietrich, C. R., & Newsam, G. N. (1997). Fast and exact simulation of
stationary Gaussian processes through circulant embedding of the covariance
matrix. SIAM Journal on Scientific Computing, 18, 1088--1107.
https://doi.org/10.1137/S1064827592240555
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from CP import estimate_hurst_cp
from fBm_circulant_embedding import circulant_fBm


DEFAULT_H_VALUES = np.array([0.2, 0.5, 0.8])
DEFAULT_TAU_VALUES = np.arange(3, 21, dtype=int)


def generate_tau_stability_data(
    *,
    H_values: np.ndarray = DEFAULT_H_VALUES,
    tau_values: np.ndarray = DEFAULT_TAU_VALUES,
    L: int = 1000,
    n_profiles: int = 10_000,
    c_fBm: float = 1.0,
    seed: int = 20260901,
) -> list[dict[str, float | int]]:
    """Generate CP estimation statistics for several values of ``tau``.

    Parameters
    ----------
    H_values : ndarray
        Hurst parameters used to generate the fBm profiles.
    tau_values : ndarray
        Positive integer CP scale parameters.
    L : int, default=1000
        Number of points in every profile.
    n_profiles : int, default=10000
        Number of independent profiles generated for every H.
    c_fBm : float, default=1.0
        Topothesy parameter of the fBm model.
    seed : int, default=20260901
        Master seed for reproducible simulation.

    Returns
    -------
    list of dict
        One summary row for every combination of H and tau.
    """
    H_values = np.asarray(H_values, dtype=float)
    tau_values = np.asarray(tau_values, dtype=int)
    if L <= 2 * int(np.max(tau_values)):
        raise ValueError("L must be greater than twice the largest tau.")
    if n_profiles < 2:
        raise ValueError("n_profiles must be at least 2.")
    if np.any((H_values <= 0.0) | (H_values >= 1.0)):
        raise ValueError("Every H value must lie strictly between 0 and 1.")
    if np.any(tau_values < 1):
        raise ValueError("Every tau must be a positive integer.")

    child_seeds = np.random.SeedSequence(seed).spawn(H_values.size)
    rows: list[dict[str, float | int]] = []

    for H_index, (H, child_seed) in enumerate(zip(H_values, child_seeds), 1):
        rng = np.random.default_rng(child_seed)
        estimates = np.empty((n_profiles, tau_values.size), dtype=float)
        profile_index = 0
        start = time.perf_counter()

        while profile_index < n_profiles:
            _, profile_1, profile_2 = circulant_fBm(
                H=float(H),
                L=L,
                x_max=float(L - 1),
                c_fBm=c_fBm,
                rng=rng,
            )
            for profile in (profile_1, profile_2):
                for tau_index, tau in enumerate(tau_values):
                    estimates[profile_index, tau_index] = estimate_hurst_cp(
                        profile, tau=int(tau)
                    )
                profile_index += 1
                if profile_index == n_profiles:
                    break

        elapsed = time.perf_counter() - start
        errors = estimates - H
        for tau_index, tau in enumerate(tau_values):
            values = estimates[:, tau_index]
            error_values = errors[:, tau_index]
            mean_estimate = float(np.mean(values))
            standard_deviation = float(np.std(values, ddof=1))
            rows.append(
                {
                    "H": float(H),
                    "L": L,
                    "tau": int(tau),
                    "n_profiles": n_profiles,
                    "c_fBm": c_fBm,
                    "mean_estimate": mean_estimate,
                    "signed_estimation_error": mean_estimate - float(H),
                    "absolute_estimation_error": abs(mean_estimate - float(H)),
                    "mean_absolute_profile_error": float(
                        np.mean(np.abs(error_values))
                    ),
                    "standard_deviation": standard_deviation,
                    "standard_error_of_mean": standard_deviation
                    / np.sqrt(n_profiles),
                    "elapsed_seconds_for_H": elapsed,
                }
            )
        print(
            f"[{H_index}/{H_values.size}] H={H:.1f}: "
            f"{n_profiles} profiles in {elapsed:.2f} s",
            flush=True,
        )
    return rows


def write_csv(rows: list[dict[str, float | int]], output_path: Path) -> None:
    """Write the generated summary statistics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_tau_stability(
    rows: list[dict[str, float | int]],
    output_stem: Path,
) -> None:
    """Create publication-quality PDF and PNG representations of the data."""
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    H_values = sorted({float(row["H"]) for row in rows})
    # Okabe--Ito-inspired high-contrast colours; blue, orange, and magenta
    # remain distinguishable in common colour-vision deficiencies.
    contrast_colors = ("#0072B2", "#E69F00", "#CC79A7")
    color_by_H = dict(zip(H_values, contrast_colors))

    figure, axis = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
    for H in H_values:
        selected = sorted(
            (row for row in rows if float(row["H"]) == H),
            key=lambda row: int(row["tau"]),
        )
        tau = np.array([int(row["tau"]) for row in selected])
        mean_error = np.array(
            [float(row["signed_estimation_error"]) for row in selected]
        )
        standard_deviation = np.array(
            [float(row["standard_deviation"]) for row in selected]
        )
        color = color_by_H[H]
        axis.fill_between(
            tau,
            mean_error - standard_deviation,
            mean_error + standard_deviation,
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        axis.plot(
            tau,
            mean_error,
            color=color,
            marker="o",
            markersize=4.0,
            label=rf"$H={H:.1f}$",
        )

    axis.axhline(0.0, color="0.25", linewidth=0.8, linestyle="--", zorder=0)
    axis.set_xticks(np.arange(3, 21, 1))
    axis.set_xlim(2.7, 20.3)
    axis.grid(True, which="major", color="0.88", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.set_xlabel(r"CP scale parameter, $\tau$")
    axis.set_ylabel(r"Mean estimation error, $\overline{\widehat{H}}-H$")
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=len(H_values),
        handlelength=2.2,
        columnspacing=2.2,
    )
    figure.suptitle(
        r"CP scale dependence ($L=1000$, $N=10\,000$ profiles)",
        fontsize=10,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=500, bbox_inches="tight")
    plt.close(figure)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-profiles", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    """Generate the data, CSV file, metadata, and scientific figure."""
    args = _parse_arguments()
    total_start = time.perf_counter()
    rows = generate_tau_stability_data(
        n_profiles=args.n_profiles,
        seed=args.seed,
    )
    total_elapsed = time.perf_counter() - total_start

    csv_path = args.output_dir / "cp_tau_stability_data.csv"
    figure_stem = args.output_dir / "cp_tau_stability"
    write_csv(rows, csv_path)
    plot_tau_stability(rows, figure_stem)

    metadata = {
        "H_values": DEFAULT_H_VALUES.tolist(),
        "L": 1000,
        "tau_values": DEFAULT_TAU_VALUES.tolist(),
        "n_profiles": args.n_profiles,
        "c_fBm": 1.0,
        "seed": args.seed,
        "total_elapsed_seconds": total_elapsed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "uncertainty_band": "mean signed error plus/minus one sample SD",
    }
    (args.output_dir / "cp_tau_stability.metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote data to {csv_path}")
    print(f"Wrote figures to {figure_stem.with_suffix('.pdf')} and .png")
    print(f"Total runtime: {total_elapsed:.2f} s")


if __name__ == "__main__":
    main()
