"""Generate HFD statistics for all supplementary parameter constellations."""

from pathlib import Path
import numpy as np

from HFD import higuchi_fd
from parameter_evaluation_common import (
    H_VALUES,
    L_VALUES,
    is_suitable_configuration,
    run_parameter_evaluation,
)


def hfd_configurations(L: int) -> list[dict]:
    """Return suitable HFD combinations from Table S1."""
    configurations = []
    for n_HFD in range(3, 21):
        for l_HFD in range(3, 16):
            for f_HFD in np.round(np.arange(0.03, 0.51, 0.01), 2):
                if is_suitable_configuration(L, n_HFD, l_HFD, float(f_HFD)):
                    configurations.append(
                        {"n": n_HFD, "l": l_HFD, "f": float(f_HFD)}
                    )
    return configurations


def _estimate_hfd(profile: np.ndarray, configuration: dict) -> float:
    return higuchi_fd(
        profile,
        n_HFD=configuration["n"],
        l_HFD=configuration["l"],
        f_HFD=configuration["f"],
    )[1]


def generate_hfd_parameter_results(
    output_path: str | Path = "hfd_parameter_results.csv",
    number_of_profiles: int = 10_000,
    H_values: tuple[float, ...] = H_VALUES,
    L_values: tuple[int, ...] = L_VALUES,
    c_fBm: float = 1.0,
    master_seed: int = 20260902,
    n_workers: int = 1,
    configuration_limit: int | None = None,
) -> list[dict]:
    """Generate raw HFD parameter statistics without selecting an optimum."""
    return run_parameter_evaluation(
        method="HFD", configuration_factory=hfd_configurations,
        estimator=_estimate_hfd, output_path=output_path,
        number_of_profiles=number_of_profiles, H_values=H_values,
        L_values=L_values, c_fBm=c_fBm, master_seed=master_seed,
        n_workers=n_workers, configuration_limit=configuration_limit,
    )


if __name__ == "__main__":
    generate_hfd_parameter_results()
