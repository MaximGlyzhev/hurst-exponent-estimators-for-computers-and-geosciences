"""Generate all statistics used in the estimator-comparison figure.

References
----------
Shevtsova, I. G. (2011). On the absolute constants in the Berry--Esseen type
inequalities for identically distributed summands. arXiv:1111.6554.
https://arxiv.org/abs/1111.6554

Method-specific references are documented in the estimator modules.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import csv
import os
import time
from datetime import datetime, timezone

import numpy as np
from scipy.special import ndtr

from CP import estimate_hurst_cp
from DFA import dfa
from HFD import higuchi_fd
from PSD import psd
from SVA import sva
from fBm_circulant_embedding import circulant_fBm


METHODS = ("CP", "DFA", "HFD", "PSD", "SVA")
H_VALUES = tuple(np.round(np.arange(0.1, 1.0, 0.1), 1))
L_VALUES = (1000, 2000, 5000)
BOUND_SETTINGS = ((0.10, 0.05), (0.10, 0.005), (0.05, 0.005))
C_BE = 0.4748


def _bound_column(p: float, epsilon: float) -> str:
    return f"BEB_p{p:g}_epsilon{epsilon:g}"


def berry_esseen_bound(
    epsilon: float,
    p: float,
    sigma: float,
    rho: float,
    C_BE: float = C_BE,
) -> int:
    """Return the smallest integer satisfying the paper's BEB condition.

    Parameters
    ----------
    epsilon : float
        Required absolute precision of the sample mean.
    p : float
        Maximum error probability; the confidence level is ``1-p``.
    sigma : float
        Standard deviation of the estimator.
    rho : float
        Absolute third central moment of the estimator.
    C_BE : float, default=0.4748
        Absolute constant in the Berry--Esseen inequality.

    Returns
    -------
    int
        Smallest sufficient number of independent profiles.

    Raises
    ------
    ValueError
        If an argument is outside its admissible range.
    OverflowError
        If the bound exceeds the range of a signed 64-bit integer.

    Notes
    -----
    This implements Eq. (supp-BEB-definition) of the Supplementary Material.
    """
    if epsilon <= 0 or not 0 < p < 1:
        raise ValueError("Require epsilon > 0 and 0 < p < 1.")
    if sigma < 0 or rho < 0:
        raise ValueError("sigma and rho must be non-negative.")
    if sigma == 0:
        if rho == 0:
            return 1
        raise ValueError("rho must be zero when sigma is zero.")

    def condition(n: int) -> bool:
        sqrt_n = np.sqrt(n)
        left = (
            1.0 - ndtr(epsilon * sqrt_n / sigma)
            + C_BE * rho / (sigma**3 * sqrt_n)
        )
        return left <= p / 2.0

    if condition(1):
        return 1
    lower, upper = 1, 2
    while not condition(upper):
        lower, upper = upper, 2 * upper
        if upper > np.iinfo(np.int64).max // 2:
            raise OverflowError("Berry--Esseen bound exceeds int64 range.")
    while lower + 1 < upper:
        middle = (lower + upper) // 2
        if condition(middle):
            upper = middle
        else:
            lower = middle
    return upper


def _estimate_all_methods(profile: np.ndarray) -> np.ndarray:
    """Apply all methods using the optimised manuscript parameters."""
    L = len(profile)
    return np.array(
        [
            estimate_hurst_cp(profile, tau=3),
            dfa(profile, l_DFA=10, f_DFA=0.1, n_DFA=20)[0],
            higuchi_fd(profile, l_HFD=3, f_HFD=0.05, n_HFD=10)[1],
            psd(
                profile,
                l_Welch=L // 2,
                O_Welch=L // 4,
                F_Welch=L,
                window="hann",
                detrend="constant",
                average="median",
                scaling="density",
                nu_min=4 / L,
                nu_max=0.05,
            )[0],
            sva(profile, l_SVA=3, f_SVA=0.05, n_SVA=10)[0],
        ],
        dtype=float,
    )


def _simulate_configuration(task: tuple) -> list[dict]:
    configuration_start = time.perf_counter()
    H, L, number_of_profiles, c_fBm, seed = task
    rng = np.random.default_rng(seed)
    estimates = np.empty((number_of_profiles, len(METHODS)))

    # Each circulant embedding provides two independent fBm profiles.
    row = 0
    while row < number_of_profiles:
        _, profile_1, profile_2 = circulant_fBm(
            H=H, L=L, x_max=L - 1, c_fBm=c_fBm, rng=rng
        )
        for profile in (profile_1, profile_2):
            if row == number_of_profiles:
                break
            estimates[row] = _estimate_all_methods(profile)
            row += 1

    if not np.all(np.isfinite(estimates)):
        index = np.argwhere(~np.isfinite(estimates))[0]
        raise RuntimeError(
            f"Non-finite result for H={H}, L={L}, "
            f"profile={index[0]}, method={METHODS[index[1]]}."
        )

    rows = []
    for method_order, method in enumerate(METHODS):
        values = estimates[:, method_order]
        mean = float(np.mean(values))
        sigma = float(np.std(values, ddof=1))
        rho = float(np.mean(np.abs(values - mean) ** 3))
        result = {
            "method": method,
            "method_order": method_order,
            "H_true": H,
            "L": L,
            "number_of_profiles": number_of_profiles,
            "mean_estimate": mean,
            "signed_estimation_error": mean - H,
            "absolute_estimation_error": abs(mean - H),
            "standard_deviation": sigma,
            "absolute_third_central_moment": rho,
            "seed": seed,
        }
        for p, epsilon in BOUND_SETTINGS:
            result[_bound_column(p, epsilon)] = berry_esseen_bound(
                epsilon, p, sigma, rho
            )
        rows.append(result)
    configuration_runtime = time.perf_counter() - configuration_start
    for result in rows:
        result["configuration_runtime_seconds"] = configuration_runtime
    return rows


def generate_estimator_comparison_results(
    output_path: str | Path = "estimator_comparison_results.csv",
    number_of_profiles: int = 10_000,
    H_values: tuple[float, ...] = H_VALUES,
    L_values: tuple[int, ...] = L_VALUES,
    c_fBm: float = 1.0,
    master_seed: int = 20260901,
    n_workers: int | None = None,
) -> list[dict]:
    """Generate and save the numerical data underlying the comparison plot.

    Parameters
    ----------
    output_path : path-like, default="estimator_comparison_results.csv"
        Destination of the self-contained CSV result table.
    number_of_profiles : int, default=10000
        Independent fBm profiles generated for every ``(H, L)`` combination.
    H_values : tuple of float
        Hurst parameters used for fBm generation. Universal bounds are maxima
        over precisely this set.
    L_values : tuple of int
        Numbers of profile points.
    c_fBm : float, default=1.0
        fBm amplitude parameter. It does not affect the five H estimates for
        unmodified profiles.
    master_seed : int, default=20260901
        Seed from which deterministic task-specific seeds are derived.
    n_workers : int, optional
        Number of worker processes. Defaults to one less than the available
        logical CPU count. Use one for sequential execution.

    Returns
    -------
    list of dict
        Rows written to the CSV. Each row contains method, ``H_true``, ``L``,
        empirical mean, signed and absolute estimation errors, sample standard
        deviation, absolute third central moment, configuration-specific BEBs,
        universal BEBs, universal total-profile-point bounds, per-configuration
        runtime, total wall-clock runtime, worker count, and UTC timestamps.

    Raises
    ------
    ValueError
        If fewer than two profiles are requested or a parameter is invalid.
    RuntimeError
        If an estimator produces a non-finite result.

    Notes
    -----
    This function generates the data for Figure 2 of the accompanying paper.
    Universal bounds are repeated over H so the single CSV is self-contained.
    Parallel execution changes task scheduling but not task-specific seeds.
    Runtime values are wall-clock measurements obtained with
    :func:`time.perf_counter` and therefore describe the current machine and
    execution settings rather than an algorithm-independent performance value.
    """
    if number_of_profiles < 2:
        raise ValueError("number_of_profiles must be at least 2.")
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    run_start = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    configurations = [(float(H), int(L)) for H in H_values for L in L_values]
    sequences = np.random.SeedSequence(master_seed).spawn(len(configurations))
    tasks = [
        (H, L, number_of_profiles, c_fBm, int(seq.generate_state(1)[0]))
        for (H, L), seq in zip(configurations, sequences)
    ]

    if n_workers == 1:
        grouped_rows = map(_simulate_configuration, tasks)
        rows = [row for group in grouped_rows for row in group]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            grouped_rows = executor.map(_simulate_configuration, tasks)
            rows = [row for group in grouped_rows for row in group]

    universal_values = {}
    for p, epsilon in BOUND_SETTINGS:
        bound = _bound_column(p, epsilon)
        universal = f"universal_{bound}"
        total_points = f"universal_profile_points_p{p:g}_epsilon{epsilon:g}"
        for method in METHODS:
            for L in L_values:
                value = max(
                    row[bound]
                    for row in rows
                    if row["method"] == method and row["L"] == L
                )
                universal_values[(method, L, universal)] = value
        for row in rows:
            value = universal_values[(row["method"], row["L"], universal)]
            row[universal] = value
            row[total_points] = value * row["L"]

    results = sorted(
        rows, key=lambda row: (row["H_true"], row["L"], row["method_order"])
    )
    total_runtime = time.perf_counter() - run_start
    finished_at = datetime.now(timezone.utc).isoformat()
    for row in results:
        row["total_runtime_seconds"] = total_runtime
        row["n_workers"] = n_workers
        row["run_started_at_utc"] = started_at
        row["run_finished_at_utc"] = finished_at
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return results


if __name__ == "__main__":
    generate_estimator_comparison_results()
