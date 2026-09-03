"""Generate PSD statistics for the fixed Welch configuration in the paper."""

from pathlib import Path
import numpy as np

from PSD import psd
from parameter_evaluation_common import H_VALUES, L_VALUES, run_parameter_evaluation


def psd_configurations(L: int) -> list[dict]:
    """Return the fixed, L-dependent Welch configuration."""
    return [{
        "l_Welch": L // 2,
        "O_Welch": L // 4,
        "F_Welch": L,
        "nu_min": 4 / L,
        "nu_max": 0.05,
    }]


def _estimate_psd(profile: np.ndarray, configuration: dict) -> float:
    return psd(
        profile,
        l_Welch=configuration["l_Welch"],
        O_Welch=configuration["O_Welch"],
        F_Welch=configuration["F_Welch"],
        window="hann", detrend="constant", average="median",
        scaling="density", nu_min=configuration["nu_min"],
        nu_max=configuration["nu_max"],
    )[0]


def generate_psd_parameter_results(
    output_path: str | Path = "psd_parameter_results.csv",
    number_of_profiles: int = 10_000,
    H_values: tuple[float, ...] = H_VALUES,
    L_values: tuple[int, ...] = L_VALUES,
    c_fBm: float = 1.0,
    root_seed: int = 20260902,
    n_workers: int = 1,
    configuration_limit: int | None = None,
) -> list[dict]:
    """Generate PSD moments for every requested ``(H, L)`` combination."""
    return run_parameter_evaluation(
        method="PSD", configuration_factory=psd_configurations,
        estimator=_estimate_psd, output_path=output_path,
        number_of_profiles=number_of_profiles, H_values=H_values,
        L_values=L_values, c_fBm=c_fBm, root_seed=root_seed,
        n_workers=n_workers, configuration_limit=configuration_limit,
    )


if __name__ == "__main__":
    generate_psd_parameter_results()
