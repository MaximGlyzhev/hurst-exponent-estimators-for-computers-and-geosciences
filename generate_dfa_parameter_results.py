"""Generate DFA statistics for all supplementary parameter constellations."""

from pathlib import Path
import numpy as np

from DFA import dfa
from parameter_evaluation_common import (
    H_VALUES,
    L_VALUES,
    is_suitable_configuration,
    run_parameter_evaluation,
)


def dfa_configurations(L: int) -> list[dict]:
    """Return suitable DFA combinations from Table S1."""
    configurations = []
    for n_DFA in range(3, 21):
        for l_DFA in range(10, 21):
            for f_DFA in np.round(np.arange(0.03, 0.51, 0.01), 2):
                if is_suitable_configuration(L, n_DFA, l_DFA, float(f_DFA)):
                    configurations.append(
                        {"n": n_DFA, "l": l_DFA, "f": float(f_DFA)}
                    )
    return configurations


def _estimate_dfa(profile: np.ndarray, configuration: dict) -> float:
    return dfa(
        profile,
        n_DFA=configuration["n"],
        l_DFA=configuration["l"],
        f_DFA=configuration["f"],
    )[0]


def generate_dfa_parameter_results(
    output_path: str | Path = "dfa_parameter_results.csv",
    number_of_profiles: int = 10_000,
    H_values: tuple[float, ...] = H_VALUES,
    L_values: tuple[int, ...] = L_VALUES,
    c_fBm: float = 1.0,
    master_seed: int = 20260902,
    n_workers: int = 1,
    configuration_limit: int | None = None,
) -> list[dict]:
    """Generate raw DFA parameter statistics without selecting an optimum."""
    return run_parameter_evaluation(
        method="DFA", configuration_factory=dfa_configurations,
        estimator=_estimate_dfa, output_path=output_path,
        number_of_profiles=number_of_profiles, H_values=H_values,
        L_values=L_values, c_fBm=c_fBm, master_seed=master_seed,
        n_workers=n_workers, configuration_limit=configuration_limit,
    )


if __name__ == "__main__":
    generate_dfa_parameter_results()
