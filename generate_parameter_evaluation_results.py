"""Deprecated combined generator for the estimator parameter study.

Use the five ``generate_<method>_parameter_results.py`` modules instead. This
file is retained temporarily for traceability and must not be used for new
production runs.

Method-specific references are documented in the estimator modules.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import csv
import os
import time

import numpy as np

from CP import estimate_hurst_cp
from DFA import dfa
from HFD import higuchi_fd
from PSD import psd
from SVA import sva
from fBm_circulant_embedding import circulant_fBm


METHODS = ("CP", "DFA", "HFD", "PSD", "SVA")
H_VALUES = tuple(np.round(np.arange(0.1, 1.0, 0.1), 1))
L_VALUES = (1000, 2000, 5000)


def _logarithmic_scales(L: int, n: int, l: int, f: float) -> np.ndarray:
    """Return the integer scales from Eq. (logarithmic-scales)."""
    raw = l * ((f * L) / l) ** (np.arange(n, dtype=float) / (n - 1))
    return np.floor(raw + 1e-12).astype(int)


def _is_suitable_configuration(L: int, n: int, l: int, f: float) -> bool:
    """Check whether rounding produces ``n`` distinct increasing scales."""
    if f * L < l:
        return False
    scales = _logarithmic_scales(L, n, l, f)
    return scales[0] >= 1 and scales[-1] < L and np.all(np.diff(scales) > 0)


def parameter_configurations(
    L: int,
    methods: tuple[str, ...] = METHODS,
) -> list[dict]:
    """Construct all parameter constellations tested in the supplement.

    Parameters
    ----------
    L : int
        Profile length for which discrete scale suitability is checked.
    methods : tuple of str
        Methods to include. CP and PSD each contribute their fixed paper
        configuration; DFA, HFD, and SVA contribute the suitable Cartesian
        products from Table S1.

    Returns
    -------
    list of dict
        Parameter records in deterministic method, ``n``, ``l``, ``f`` order.

    Raises
    ------
    ValueError
        If an unknown method is requested.
    """
    unknown = set(methods).difference(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")

    configurations = []
    if "CP" in methods:
        configurations.append({"method": "CP", "tau": 3})

    ranges = {
        "DFA": (range(3, 21), range(10, 21)),
        "HFD": (range(3, 21), range(3, 16)),
        "SVA": (range(3, 21), range(3, 16)),
    }
    f_values = np.round(np.arange(0.03, 0.51, 0.01), 2)
    for method in ("DFA", "HFD", "SVA"):
        if method not in methods:
            continue
        n_values, l_values = ranges[method]
        for n in n_values:
            for l in l_values:
                for f in f_values:
                    if _is_suitable_configuration(L, n, l, float(f)):
                        configurations.append(
                            {"method": method, "n": n, "l": l, "f": float(f)}
                        )

    if "PSD" in methods:
        configurations.append(
            {
                "method": "PSD",
                "l_Welch": L // 2,
                "O_Welch": L // 4,
                "F_Welch": L,
                "nu_min": 4 / L,
                "nu_max": 0.05,
            }
        )
    return configurations


def _estimate(profile: np.ndarray, configuration: dict) -> float:
    method = configuration["method"]
    if method == "CP":
        return estimate_hurst_cp(profile, tau=configuration["tau"])
    if method == "DFA":
        return dfa(
            profile,
            l_DFA=configuration["l"],
            f_DFA=configuration["f"],
            n_DFA=configuration["n"],
        )[0]
    if method == "HFD":
        return higuchi_fd(
            profile,
            l_HFD=configuration["l"],
            f_HFD=configuration["f"],
            n_HFD=configuration["n"],
        )[1]
    if method == "PSD":
        return psd(
            profile,
            l_Welch=configuration["l_Welch"],
            O_Welch=configuration["O_Welch"],
            F_Welch=configuration["F_Welch"],
            window="hann",
            detrend="constant",
            average="median",
            scaling="density",
            nu_min=configuration["nu_min"],
            nu_max=configuration["nu_max"],
        )[0]
    if method == "SVA":
        return sva(
            profile,
            l_SVA=configuration["l"],
            f_SVA=configuration["f"],
            n_SVA=configuration["n"],
        )[0]
    raise ValueError(f"Unknown method: {method}")


def _generate_profiles(
    H: float,
    L: int,
    number_of_profiles: int,
    c_fBm: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    profiles = np.empty((number_of_profiles, L), dtype=float)
    row = 0
    while row < number_of_profiles:
        _, profile_1, profile_2 = circulant_fBm(
            H=H, L=L, x_max=L - 1, c_fBm=c_fBm, rng=rng
        )
        for profile in (profile_1, profile_2):
            if row == number_of_profiles:
                break
            profiles[row] = profile
            row += 1
    return profiles


def _evaluate_H_L(task: tuple) -> tuple[list[dict], float]:
    (
        H, L, number_of_profiles, c_fBm, seed, methods,
        configuration_limit,
    ) = task
    task_start = time.perf_counter()
    profiles = _generate_profiles(H, L, number_of_profiles, c_fBm, seed)
    configurations = parameter_configurations(L, methods=methods)
    if configuration_limit is not None:
        configurations = configurations[:configuration_limit]

    rows = []
    for configuration_order, configuration in enumerate(configurations):
        configuration_start = time.perf_counter()
        estimates = np.fromiter(
            (_estimate(profile, configuration) for profile in profiles),
            dtype=float,
            count=number_of_profiles,
        )
        if not np.all(np.isfinite(estimates)):
            raise RuntimeError(
                f"Non-finite estimate for H={H}, L={L}, "
                f"configuration={configuration}."
            )
        mean = float(np.mean(estimates))
        row = {
            "method": configuration["method"],
            "configuration_order": configuration_order,
            "H_true": H,
            "L": L,
            "number_of_profiles": number_of_profiles,
            "mean_estimate": mean,
            "signed_estimation_error": mean - H,
            "absolute_estimation_error": abs(mean - H),
            "standard_deviation": float(np.std(estimates, ddof=1)),
            "absolute_third_central_moment": float(
                np.mean(np.abs(estimates - mean) ** 3)
            ),
            "seed": seed,
            "configuration_runtime_seconds": time.perf_counter()
            - configuration_start,
            "tau": "",
            "n": "",
            "l": "",
            "f": "",
            "l_Welch": "",
            "O_Welch": "",
            "F_Welch": "",
            "nu_min": "",
            "nu_max": "",
        }
        row.update(configuration)
        rows.append(row)
    return rows, time.perf_counter() - task_start


def generate_parameter_evaluation_results(
    output_path: str | Path = "parameter_evaluation_results.csv",
    number_of_profiles: int = 10_000,
    H_values: tuple[float, ...] = H_VALUES,
    L_values: tuple[int, ...] = L_VALUES,
    methods: tuple[str, ...] = METHODS,
    c_fBm: float = 1.0,
    master_seed: int = 20260902,
    n_workers: int = 1,
    configuration_limit: int | None = None,
) -> list[dict]:
    """Generate raw statistics for all tested parameter configurations.

    Parameters
    ----------
    output_path : path-like, default="parameter_evaluation_results.csv"
        Destination of the raw result table.
    number_of_profiles : int, default=10000
        Profiles evaluated for each ``(H, L)`` combination.
    H_values, L_values : tuple
        fBm Hurst parameters and profile lengths.
    methods : tuple of str
        Subset of ``CP, DFA, HFD, PSD, SVA`` to evaluate.
    c_fBm : float, default=1.0
        fBm amplitude parameter.
    master_seed : int, default=20260902
        Root seed used to derive one deterministic seed per ``(H, L)``.
    n_workers : int, default=1
        Parallel ``(H, L)`` tasks. The conservative default limits memory use;
        one L=5000 task stores about 400 MB for 10000 float64 profiles.
    configuration_limit : int, optional
        Test-only limit applied after configuration generation. Leave ``None``
        for the complete supplementary parameter space.

    Returns
    -------
    list of dict
        Raw rows also written to the CSV. No optimisation or aggregation over
        H is performed.

    Raises
    ------
    ValueError
        If arguments or requested methods are invalid.
    RuntimeError
        If an estimator returns a non-finite value.

    Notes
    -----
    The function reuses exactly the same generated profiles across all
    parameter constellations for a fixed ``(H, L)``. The complete exhaustive
    study is computationally very expensive. ``configuration_limit`` exists
    only for smoke tests and must not be used for the published data.
    """
    if number_of_profiles < 2:
        raise ValueError("number_of_profiles must be at least 2.")
    if n_workers < 1:
        raise ValueError("n_workers must be at least 1.")
    parameter_configurations(int(L_values[0]), methods=methods)

    run_start = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    combinations = [(float(H), int(L)) for H in H_values for L in L_values]
    sequences = np.random.SeedSequence(master_seed).spawn(len(combinations))
    tasks = [
        (
            H, L, number_of_profiles, c_fBm,
            int(sequence.generate_state(1)[0]), methods, configuration_limit,
        )
        for (H, L), sequence in zip(combinations, sequences)
    ]

    if n_workers == 1:
        outputs = map(_evaluate_H_L, tasks)
        evaluated = list(outputs)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            evaluated = list(executor.map(_evaluate_H_L, tasks))

    rows = []
    for (H, L), (task_rows, task_runtime) in zip(combinations, evaluated):
        for row in task_rows:
            row["H_L_runtime_seconds"] = task_runtime
            rows.append(row)

    total_runtime = time.perf_counter() - run_start
    finished_at = datetime.now(timezone.utc).isoformat()
    for row in rows:
        row["total_runtime_seconds"] = total_runtime
        row["n_workers"] = n_workers
        row["run_started_at_utc"] = started_at
        row["run_finished_at_utc"] = finished_at

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No parameter configurations were evaluated.")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    generate_parameter_evaluation_results()
