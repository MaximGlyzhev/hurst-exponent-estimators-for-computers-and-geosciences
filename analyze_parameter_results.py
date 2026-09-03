"""Aggregate estimator parameter studies and validate manuscript choices.

The script reads the method-specific CSV files produced by the parameter
generators. For each method, profile length, and tuning-parameter
configuration, it computes worst-case statistics over the simulated Hurst
parameters. It identifies the two-objective Pareto frontier and marks the
parameter configurations explicitly used in the manuscript.

No undocumented scalar optimisation objective is introduced. In particular,
the DFA choice represents the documented trade-off after excluding
configurations whose maximum absolute estimation error exceeds 0.05.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr


METHOD_FILES = {
    "CP": "cp_parameter_results.csv",
    "DFA": "dfa_parameter_results.csv",
    "HFD": "hfd_parameter_results.csv",
    "PSD": "psd_parameter_results.csv",
    "SVA": "sva_parameter_results.csv",
}
CONFIGURATION_FIELDS = {
    "CP": ("tau",),
    "DFA": ("n", "l", "f"),
    "HFD": ("n", "l", "f"),
    "PSD": ("l_Welch", "O_Welch", "F_Welch", "nu_min", "nu_max"),
    "SVA": ("n", "l", "f"),
}
OUTPUT_FIELDS = (
    "method", "L", "number_of_H_values", "number_of_profiles_per_H",
    "max_absolute_estimation_error", "max_standard_deviation",
    "max_absolute_third_central_moment", "mean_absolute_estimation_error",
    "universal_BEB_profiles_p0.05_epsilon0.005",
    "pareto_optimal", "passes_DFA_error_threshold",
    "minimum_DFA_BEB_among_eligible", "manuscript_choice",
    "tau", "n", "l", "f", "l_Welch", "O_Welch", "F_Welch",
    "nu_min", "nu_max",
)
C_BE = 0.4748


def berry_esseen_bound(
    epsilon: float,
    p: float,
    sigma: float,
    rho: float,
) -> int:
    """Return the smallest sample size satisfying the paper's BE condition."""
    if epsilon <= 0.0 or not 0.0 < p < 1.0:
        raise ValueError("Require epsilon > 0 and 0 < p < 1.")
    if sigma < 0.0 or rho < 0.0:
        raise ValueError("sigma and rho must be non-negative.")
    if sigma == 0.0:
        if rho == 0.0:
            return 1
        raise ValueError("rho must be zero when sigma is zero.")

    def condition(number_of_profiles: int) -> bool:
        sqrt_n = np.sqrt(number_of_profiles)
        left_hand_side = (
            1.0 - ndtr(epsilon * sqrt_n / sigma)
            + C_BE * rho / (sigma**3 * sqrt_n)
        )
        return bool(left_hand_side <= p / 2.0)

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


def _number(value: str) -> int | float | str:
    """Convert a non-empty CSV value to an integer or float when possible."""
    if value == "":
        return ""
    try:
        numeric = float(value)
    except ValueError:
        return value
    return int(numeric) if numeric.is_integer() else numeric


def read_method_results(path: Path, expected_method: str) -> list[dict]:
    """Read and validate one method-specific raw result file."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No result rows found in {path}.")
    required = {
        "method", "H_true", "L", "number_of_profiles",
        "absolute_estimation_error", "standard_deviation",
        "absolute_third_central_moment",
    } | set(CONFIGURATION_FIELDS[expected_method])
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    converted = []
    for row in rows:
        if row["method"] != expected_method:
            raise ValueError(
                f"Expected method {expected_method} in {path}, found {row['method']}."
            )
        converted.append({key: _number(value) for key, value in row.items()})
    return converted


def _configuration_key(method: str, row: dict) -> tuple:
    return tuple(row[field] for field in CONFIGURATION_FIELDS[method])


def _is_manuscript_choice(method: str, row: dict) -> bool:
    """Return whether a summary row matches the parameters used in the paper."""
    def close(field: str, target: float) -> bool:
        return bool(np.isclose(float(row[field]), target, rtol=0.0, atol=1e-12))

    if method == "CP":
        return close("tau", 3)
    if method == "DFA":
        return close("n", 20) and close("l", 10) and close("f", 0.1)
    if method in {"HFD", "SVA"}:
        return close("n", 10) and close("l", 3) and close("f", 0.05)
    if method == "PSD":
        # PSD has one fixed L-dependent configuration in its generator.
        return True
    return False


def aggregate_method(method: str, rows: list[dict]) -> list[dict]:
    """Aggregate raw rows over H for each L and parameter configuration."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(int(row["L"]), _configuration_key(method, row))].append(row)

    summaries = []
    for (L, configuration), members in groups.items():
        result = {field: "" for field in OUTPUT_FIELDS}
        result.update({"method": method, "L": L})
        result.update(dict(zip(CONFIGURATION_FIELDS[method], configuration)))
        errors = np.array([float(row["absolute_estimation_error"]) for row in members])
        deviations = np.array([float(row["standard_deviation"]) for row in members])
        moments = np.array(
            [float(row["absolute_third_central_moment"]) for row in members]
        )
        result.update(
            {
                "number_of_H_values": len({float(row["H_true"]) for row in members}),
                "number_of_profiles_per_H": int(members[0]["number_of_profiles"]),
                "max_absolute_estimation_error": float(np.max(errors)),
                "max_standard_deviation": float(np.max(deviations)),
                "max_absolute_third_central_moment": float(np.max(moments)),
                "mean_absolute_estimation_error": float(np.mean(errors)),
                "passes_DFA_error_threshold": (
                    method != "DFA" or float(np.max(errors)) <= 0.05
                ),
                "universal_BEB_profiles_p0.05_epsilon0.005": max(
                    berry_esseen_bound(
                        epsilon=0.005,
                        p=0.05,
                        sigma=float(row["standard_deviation"]),
                        rho=float(row["absolute_third_central_moment"]),
                    )
                    for row in members
                ),
                "minimum_DFA_BEB_among_eligible": False,
            }
        )
        result["manuscript_choice"] = _is_manuscript_choice(method, result)
        summaries.append(result)

    # A configuration is Pareto optimal if no other configuration at the same
    # method and L is at least as good in both maximum error and maximum SD and
    # strictly better in one.
    for L in sorted({int(row["L"]) for row in summaries}):
        candidates = [row for row in summaries if int(row["L"]) == L]
        for candidate in candidates:
            error = float(candidate["max_absolute_estimation_error"])
            deviation = float(candidate["max_standard_deviation"])
            candidate["pareto_optimal"] = not any(
                (
                    float(other["max_absolute_estimation_error"]) <= error
                    and float(other["max_standard_deviation"]) <= deviation
                    and (
                        float(other["max_absolute_estimation_error"]) < error
                        or float(other["max_standard_deviation"]) < deviation
                    )
                )
                for other in candidates
            )
        if method == "DFA":
            eligible = [row for row in candidates if row["passes_DFA_error_threshold"]]
            if eligible:
                minimum_bound = min(
                    int(row["universal_BEB_profiles_p0.05_epsilon0.005"])
                    for row in eligible
                )
                for row in eligible:
                    row["minimum_DFA_BEB_among_eligible"] = (
                        int(row["universal_BEB_profiles_p0.05_epsilon0.005"])
                        == minimum_bound
                    )
    return summaries


def write_csv(rows: list[dict], path: Path) -> None:
    """Write rectangular summary records to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict], output_stem: Path) -> None:
    """Plot worst-case error against worst-case standard deviation."""
    methods = [method for method in ("DFA", "HFD", "SVA") if any(
        row["method"] == method for row in rows
    )]
    if not methods:
        return
    L_values = sorted({int(row["L"]) for row in rows})
    colors = dict(zip(L_values, ("#0072B2", "#E69F00", "#CC79A7")))
    figure, axes = plt.subplots(
        1, len(methods), figsize=(2.7 * len(methods), 2.8), squeeze=False,
        constrained_layout=True,
    )
    for axis, method in zip(axes[0], methods):
        method_rows = [row for row in rows if row["method"] == method]
        for L in L_values:
            selected = [row for row in method_rows if int(row["L"]) == L]
            if not selected:
                continue
            axis.scatter(
                [row["max_absolute_estimation_error"] for row in selected],
                [row["max_standard_deviation"] for row in selected],
                s=8, alpha=0.25, color=colors[L], label=rf"$L={L}$",
            )
            chosen = [row for row in selected if row["manuscript_choice"]]
            if chosen:
                axis.scatter(
                    [row["max_absolute_estimation_error"] for row in chosen],
                    [row["max_standard_deviation"] for row in chosen],
                    marker="*", s=100, color=colors[L], edgecolor="black",
                    linewidth=0.6, zorder=5,
                )
        axis.set_title(method)
        axis.set_xlabel("Maximum absolute estimation error")
        axis.grid(True, color="0.9", linewidth=0.6)
    axes[0, 0].set_ylabel("Maximum standard deviation")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=len(L_values), frameon=False)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=500, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Run aggregation, selection validation, and diagnostic plotting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("results/parameters"))
    parser.add_argument(
        "--allow-missing", action="store_true",
        help="Analyse available method files instead of requiring all five.",
    )
    args = parser.parse_args()

    all_summaries = []
    missing = []
    for method, filename in METHOD_FILES.items():
        path = args.input_dir / filename
        if not path.exists():
            missing.append(filename)
            continue
        all_summaries.extend(aggregate_method(method, read_method_results(path, method)))
    if missing and not args.allow_missing:
        raise FileNotFoundError(
            "Missing parameter result files: " + ", ".join(missing)
            + ". Run the corresponding generators first or use --allow-missing."
        )
    if not all_summaries:
        raise FileNotFoundError("No method-specific parameter result files were found.")

    all_summaries.sort(
        key=lambda row: (row["method"], int(row["L"]),
                         float(row["max_absolute_estimation_error"]))
    )
    selected = [row for row in all_summaries if row["manuscript_choice"]]
    write_csv(all_summaries, args.output_dir / "parameter_configuration_summary.csv")
    write_csv(selected, args.output_dir / "manuscript_parameter_choices.csv")
    plot_summary(all_summaries, args.output_dir / "parameter_configuration_summary")
    print(f"Analysed {len(all_summaries)} configurations.")
    print(f"Validated {len(selected)} manuscript selections.")
    if missing:
        print("Skipped missing files: " + ", ".join(missing))


if __name__ == "__main__":
    main()
