"""Generate the data reported in the CP missing-observation table.

For each prescribed combination of the fBm Hurst exponent ``H``, original
profile length ``L``, and missing-observation probability ``q``, independent
source profiles are generated until ``N_v * L_v`` valid CP triples are
available. The ordered triples are divided into ``N_v`` consecutive blocks
of ``L_v`` triples. A Hurst exponent is estimated from each block, after
which their arithmetic mean and sample standard deviation are reported.

The default parameters reproduce the table in the accompanying paper:

* H in {0.2, 0.5, 0.8};
* (L, q) in {(100, 0.05), (1000, 0.05), (100, 0.20), (1000, 0.20)};
* tau = 3, N_v = 10,000, and L_v = 1,000.

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

import numpy as np

from CP_gapped import valid_change_indicators
from fBm_circulant_embedding import circulant_fBm


H_VALUES = (0.2, 0.5, 0.8)
GAP_CONFIGURATIONS = (
    (100, 0.05),
    (1000, 0.05),
    (100, 0.20),
    (1000, 0.20),
)
TAU = 3
N_V = 10_000
L_V = 1_000
ROOT_SEED = 20260902


def evaluate_configuration(
    *,
    H: float,
    L: int,
    q: float,
    tau: int,
    N_v: int,
    L_v: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Evaluate one adaptive-``N`` missing-observation configuration.

    Source profiles are generated until exactly ``N_v * L_v`` ordered valid
    triples have been stored. Surplus valid triples from the final source
    profile are discarded.

    Returns
    -------
    dict
        Simulation parameters, generated source-profile count, mean Hurst
        estimate, estimation error, sample standard deviation, and standard
        error of the mean.
    """
    if not 0.0 < H < 1.0:
        raise ValueError("H must lie strictly between zero and one.")
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie strictly between zero and one.")
    if N_v < 2 or L_v < 1:
        raise ValueError("N_v must be at least 2 and L_v must be positive.")

    required_triples = N_v * L_v
    indicators = np.empty(required_triples, dtype=np.uint8)
    number_stored = 0
    N_generated = 0

    while number_stored < required_triples:
        _, first_profile, second_profile = circulant_fBm(
            H=H,
            L=L,
            x_max=float(L - 1),
            c_fBm=1.0,
            rng=rng,
        )
        for profile in (first_profile, second_profile):
            observed = rng.random(L) >= q
            current = valid_change_indicators(
                profile, tau=tau, observed=observed
            )
            number_to_store = min(current.size, required_triples - number_stored)
            indicators[number_stored : number_stored + number_to_store] = (
                current[:number_to_store]
            )
            number_stored += number_to_store
            N_generated += 1
            if number_stored == required_triples:
                break

    p_hat = indicators.reshape(N_v, L_v).mean(axis=1)
    with np.errstate(divide="raise", invalid="raise"):
        H_hat = 1.0 + np.log2(np.cos(np.pi * p_hat / 2.0))

    mean_estimate = float(np.mean(H_hat))
    standard_deviation = float(np.std(H_hat, ddof=1))
    return {
        "H": H,
        "L": L,
        "q": q,
        "tau": tau,
        "N_v": N_v,
        "L_v": L_v,
        "N_generated": N_generated,
        "valid_triples_used": required_triples,
        "mean_estimate": mean_estimate,
        "signed_estimation_error": mean_estimate - H,
        "absolute_estimation_error": abs(mean_estimate - H),
        "standard_deviation": standard_deviation,
        "standard_error_of_mean": standard_deviation / np.sqrt(N_v),
    }


def generate_table_data(
    *,
    H_values: tuple[float, ...] = H_VALUES,
    configurations: tuple[tuple[int, float], ...] = GAP_CONFIGURATIONS,
    tau: int = TAU,
    N_v: int = N_V,
    L_v: int = L_V,
    seed: int = ROOT_SEED,
) -> list[dict[str, float | int]]:
    """Generate all rows of the missing-observation results table."""
    cases = [(H, L, q) for H in H_values for L, q in configurations]
    child_seeds = np.random.SeedSequence(seed).spawn(len(cases))
    rows: list[dict[str, float | int]] = []
    for index, ((H, L, q), child_seed) in enumerate(zip(cases, child_seeds), 1):
        configuration_seed = int(child_seed.generate_state(1, dtype=np.uint64)[0])
        start = time.perf_counter()
        row = evaluate_configuration(
            H=H,
            L=L,
            q=q,
            tau=tau,
            N_v=N_v,
            L_v=L_v,
            rng=np.random.default_rng(configuration_seed),
        )
        row["seed"] = configuration_seed
        row["elapsed_seconds"] = time.perf_counter() - start
        rows.append(row)
        print(
            f"[{index}/{len(cases)}] H={H:.1f}, L={L}, q={q:.0%}: "
            f"N={row['N_generated']}, mean={row['mean_estimate']:.6f}, "
            f"SD={row['standard_deviation']:.6f}",
            flush=True,
        )
    return rows


def write_results(rows: list[dict[str, float | int]], output: Path) -> None:
    """Write table data to CSV and simulation metadata to JSON."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "description": "CP missing-observation table data",
        "H_values": H_VALUES,
        "gap_configurations": GAP_CONFIGURATIONS,
        "tau": TAU,
        "N_v": N_V,
        "L_v": L_V,
        "root_seed": ROOT_SEED,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Run the prescribed experiment and save its table data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cp_gap_table_data.csv"),
        help="Destination CSV file (default: cp_gap_table_data.csv).",
    )
    args = parser.parse_args()
    rows = generate_table_data()
    write_results(rows, args.output)
    print(f"Wrote {args.output} and {args.output.with_suffix('.metadata.json')}")


if __name__ == "__main__":
    main()
