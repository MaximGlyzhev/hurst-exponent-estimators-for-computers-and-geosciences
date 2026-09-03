"""Generate SVA statistics for all supplementary parameter constellations."""

from pathlib import Path
import numpy as np

from SVA import sva
from parameter_evaluation_common import (
    H_VALUES,
    L_VALUES,
    is_suitable_configuration,
    run_parameter_evaluation,
)


def sva_configurations(L: int) -> list[dict]:
    """Return suitable SVA combinations from Table S1."""
    configurations = []
    for n_SVA in range(3, 21):
        for l_SVA in range(3, 16):
            for f_SVA in np.round(np.arange(0.03, 0.51, 0.01), 2):
                if is_suitable_configuration(L, n_SVA, l_SVA, float(f_SVA)):
                    configurations.append(
                        {"n": n_SVA, "l": l_SVA, "f": float(f_SVA)}
                    )
    return configurations


def _estimate_sva(profile: np.ndarray, configuration: dict) -> float:
    return sva(
        profile,
        n_SVA=configuration["n"],
        l_SVA=configuration["l"],
        f_SVA=configuration["f"],
    )[0]


def generate_sva_parameter_results(
    output_path: str | Path = "sva_parameter_results.csv",
    number_of_profiles: int = 10_000,
    H_values: tuple[float, ...] = H_VALUES,
    L_values: tuple[int, ...] = L_VALUES,
    c_fBm: float = 1.0,
    root_seed: int = 20260902,
    n_workers: int = 1,
    configuration_limit: int | None = None,
) -> list[dict]:
    """Generate raw SVA parameter statistics without selecting an optimum."""
    return run_parameter_evaluation(
        method="SVA", configuration_factory=sva_configurations,
        estimator=_estimate_sva, output_path=output_path,
        number_of_profiles=number_of_profiles, H_values=H_values,
        L_values=L_values, c_fBm=c_fBm, root_seed=root_seed,
        n_workers=n_workers, configuration_limit=configuration_limit,
    )


if __name__ == "__main__":
    generate_sva_parameter_results()
