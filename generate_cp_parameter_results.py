"""Generate CP statistics for the fixed scale used in the paper."""

from pathlib import Path
import numpy as np

from CP import estimate_hurst_cp
from parameter_evaluation_common import H_VALUES, L_VALUES, run_parameter_evaluation


def cp_configurations(L: int) -> list[dict]:
    """Return the fixed CP configuration; ``L`` is accepted for consistency."""
    del L
    return [{"tau": 3}]


def _estimate_cp(profile: np.ndarray, configuration: dict) -> float:
    return estimate_hurst_cp(profile, tau=configuration["tau"])


def generate_cp_parameter_results(
    output_path: str | Path = "cp_parameter_results.csv",
    number_of_profiles: int = 10_000,
    H_values: tuple[float, ...] = H_VALUES,
    L_values: tuple[int, ...] = L_VALUES,
    c_fBm: float = 1.0,
    root_seed: int = 20260902,
    n_workers: int = 1,
    configuration_limit: int | None = None,
) -> list[dict]:
    """Generate CP moments for every requested ``(H, L)`` combination.

    See :func:`parameter_evaluation_common.run_parameter_evaluation` for the
    shared simulation, seed, runtime, and output semantics.
    """
    return run_parameter_evaluation(
        method="CP", configuration_factory=cp_configurations,
        estimator=_estimate_cp, output_path=output_path,
        number_of_profiles=number_of_profiles, H_values=H_values,
        L_values=L_values, c_fBm=c_fBm, root_seed=root_seed,
        n_workers=n_workers, configuration_limit=configuration_limit,
    )


if __name__ == "__main__":
    generate_cp_parameter_results()
