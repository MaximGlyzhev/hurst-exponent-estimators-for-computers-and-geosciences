"""Shared infrastructure for method-specific parameter evaluations.

This module contains simulation, moment estimation, deterministic seeding,
runtime measurement, and CSV writing. Method-specific parameter grids and
estimators remain in separate public generator modules.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import csv
import os
import platform
import sys
import time
from typing import Callable

import numpy as np
import scipy

from fBm_circulant_embedding import circulant_fBm


H_VALUES = tuple(np.round(np.arange(0.1, 1.0, 0.1), 1))
L_VALUES = (1000, 2000, 5000)

RESULT_FIELDS = (
    "method", "configuration_order", "H_true", "L", "number_of_profiles",
    "mean_estimate", "signed_estimation_error", "absolute_estimation_error",
    "standard_deviation", "absolute_third_central_moment", "seed",
    "configuration_runtime_seconds", "H_L_runtime_seconds",
    "total_runtime_seconds", "n_workers", "run_started_at_utc",
    "run_finished_at_utc", "tau", "n", "l", "f", "l_Welch",
    "O_Welch", "F_Welch", "nu_min", "nu_max",
)


def logarithmic_scales(L: int, n: int, l: int, f: float) -> np.ndarray:
    """Return the discrete scales from Eq. (logarithmic-scales).

    Parameters
    ----------
    L, n, l, f
        Profile length, number of scales, minimum scale, and upper cut-off
        coefficient in the notation of the paper.

    Returns
    -------
    ndarray of int
        Rounded fitting scales in increasing target order.
    """
    raw = l * ((f * L) / l) ** (np.arange(n, dtype=float) / (n - 1))
    return np.floor(raw + 1e-12).astype(int)


def is_suitable_configuration(L: int, n: int, l: int, f: float) -> bool:
    """Return whether rounding produces ``n`` valid distinct scales."""
    if n < 2 or l < 1 or f <= 0 or f * L < l:
        return False
    scales = logarithmic_scales(L, n, l, f)
    return scales[0] >= 1 and scales[-1] < L and np.all(np.diff(scales) > 0)


def generate_profiles(
    H: float,
    L: int,
    number_of_profiles: int,
    c_fBm: float,
    seed: int,
) -> np.ndarray:
    """Generate a reproducible matrix of independent fBm profiles."""
    rng = np.random.default_rng(seed)
    profiles = np.empty((number_of_profiles, L), dtype=float)
    row = 0
    while row < number_of_profiles:
        _, first, second = circulant_fBm(
            H=H, L=L, x_max=L - 1, c_fBm=c_fBm, rng=rng
        )
        for profile in (first, second):
            if row == number_of_profiles:
                break
            profiles[row] = profile
            row += 1
    return profiles


def _evaluate_H_L(task: tuple) -> tuple[list[dict], float]:
    (
        method, H, L, number_of_profiles, c_fBm, seed,
        configuration_factory, estimator, configuration_limit,
        progress_offset, total_configurations, report_each_configuration,
    ) = task
    H_L_start = time.perf_counter()
    profiles = generate_profiles(H, L, number_of_profiles, c_fBm, seed)
    configurations = configuration_factory(L)
    if configuration_limit is not None:
        configurations = configurations[:configuration_limit]

    rows = []
    for order, configuration in enumerate(configurations):
        start = time.perf_counter()
        estimates = np.fromiter(
            (estimator(profile, configuration) for profile in profiles),
            dtype=float,
            count=number_of_profiles,
        )
        if not np.all(np.isfinite(estimates)):
            raise RuntimeError(
                f"Non-finite {method} estimate for H={H}, L={L}, "
                f"configuration={configuration}."
            )
        mean = float(np.mean(estimates))
        row = {field: "" for field in RESULT_FIELDS}
        row.update(
            {
                "method": method,
                "configuration_order": order,
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
                "configuration_runtime_seconds": time.perf_counter() - start,
            }
        )
        row.update(configuration)
        rows.append(row)
        if report_each_configuration:
            completed = progress_offset + order + 1
            print(
                f"[{completed}/{total_configurations}] {method}: "
                f"H={H:g}, L={L}, parameters={configuration}",
                flush=True,
            )
    return rows, time.perf_counter() - H_L_start


def run_parameter_evaluation(
    *,
    method: str,
    configuration_factory: Callable[[int], list[dict]],
    estimator: Callable[[np.ndarray, dict], float],
    output_path: str | Path,
    number_of_profiles: int = 10_000,
    H_values: tuple[float, ...] = H_VALUES,
    L_values: tuple[int, ...] = L_VALUES,
    c_fBm: float = 1.0,
    root_seed: int = 20260902,
    n_workers: int = 1,
    configuration_limit: int | None = None,
) -> list[dict]:
    """Run and save one method-specific parameter evaluation.

    ``configuration_limit`` is intended only for smoke tests. All separate
    method generators use the same seed derivation, so equal ``root_seed``,
    ``H_values``, and ``L_values`` produce identical fBm samples.
    """
    if number_of_profiles < 2:
        raise ValueError("number_of_profiles must be at least 2.")
    if n_workers < 1:
        raise ValueError("n_workers must be at least 1.")
    if configuration_limit is not None and configuration_limit < 1:
        raise ValueError("configuration_limit must be positive or None.")

    started_at = datetime.now(timezone.utc).isoformat()
    run_start = time.perf_counter()
    combinations = [(float(H), int(L)) for H in H_values for L in L_values]
    sequences = np.random.SeedSequence(root_seed).spawn(len(combinations))
    configuration_counts = []
    for _, L in combinations:
        count = len(configuration_factory(L))
        if configuration_limit is not None:
            count = min(count, configuration_limit)
        configuration_counts.append(count)
    total_configurations = sum(configuration_counts)

    tasks = []
    progress_offset = 0
    for (H, L), sequence, count in zip(
        combinations, sequences, configuration_counts
    ):
        tasks.append(
            (
                method, H, L, number_of_profiles, c_fBm,
                int(sequence.generate_state(1)[0]), configuration_factory,
                estimator, configuration_limit, progress_offset,
                total_configurations, n_workers == 1,
            )
        )
        progress_offset += count

    if n_workers == 1:
        evaluated = list(map(_evaluate_H_L, tasks))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            evaluated = []
            completed_configurations = 0
            for task_rows, H_L_runtime in executor.map(_evaluate_H_L, tasks):
                evaluated.append((task_rows, H_L_runtime))
                completed_configurations += len(task_rows)
                if task_rows:
                    print(
                        f"[{completed_configurations}/{total_configurations}] "
                        f"{method}: completed H={task_rows[0]['H_true']:g}, "
                        f"L={task_rows[0]['L']} "
                        f"({len(task_rows)} configurations)",
                        flush=True,
                    )

    rows = []
    for task_rows, H_L_runtime in evaluated:
        for row in task_rows:
            row["H_L_runtime_seconds"] = H_L_runtime
            rows.append(row)

    total_runtime = time.perf_counter() - run_start
    finished_at = datetime.now(timezone.utc).isoformat()
    for row in rows:
        row["total_runtime_seconds"] = total_runtime
        row["n_workers"] = n_workers
        row["run_started_at_utc"] = started_at
        row["run_finished_at_utc"] = finished_at

    if not rows:
        raise RuntimeError(f"No {method} configurations were evaluated.")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    metadata_path = destination.with_suffix(".metadata.txt")
    metadata_path.write_text(
        "\n".join(
            (
                f"method={method}",
                f"result_file={destination.name}",
                f"number_of_profiles={number_of_profiles}",
                f"H_values={tuple(H_values)}",
                f"L_values={tuple(L_values)}",
                f"root_seed={root_seed}",
                f"n_workers={n_workers}",
                f"total_runtime_seconds={total_runtime}",
                f"started_at_utc={started_at}",
                f"finished_at_utc={finished_at}",
                f"python={sys.version.split()[0]}",
                f"numpy={np.__version__}",
                f"scipy={scipy.__version__}",
                f"platform={platform.platform()}",
                f"processor={platform.processor()}",
                f"logical_cpus={os.cpu_count()}",
                f"configuration_limit={configuration_limit}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return rows
