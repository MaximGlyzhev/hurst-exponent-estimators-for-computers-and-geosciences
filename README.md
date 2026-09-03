# Hurst exponent estimators and stability tests

This repository contains the Python implementation accompanying the paper
*A comparison of Hurst exponent estimators for fracture roughness analysis*.
It provides five Hurst exponent estimators, exact one-dimensional fractional
Brownian motion (fBm) simulation, parameter studies, estimator comparisons,
and stability experiments for the change-probability (CP) method.

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy
- Matplotlib
- pandas

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Estimators

The estimator modules use the notation of the paper:

| File | Method |
|---|---|
| `CP.py` | Change probability (CP) |
| `CP_gapped.py` | Modified CP estimator for regularly sampled profiles with missing observations |
| `DFA.py` | Detrended fluctuation analysis (DFA) for fBm profiles |
| `HFD.py` | Higuchi fractal dimension (HFD) |
| `PSD.py` | Power spectral density (PSD) using Welch's method |
| `SVA.py` | Second-order variogram analysis (SVA) |

For example:

```python
import numpy as np

from CP import estimate_hurst_cp
from CP_gapped import estimate_hurst_cp_gapped

z = np.loadtxt("profile.csv")
H_CP = estimate_hurst_cp(z, tau=3)

z[[12, 48, 73]] = np.nan
result = estimate_hurst_cp_gapped(z, tau=3, N_v=20, L_v=1000)
print(result.H, result.standard_deviation)
```

Missing values can be encoded as `numpy.nan` or supplied through the
`observed_masks` argument. Here, `N_v` is the number of evaluation blocks and
`L_v` is the number of valid CP triples in each block. If `N_v` is omitted,
all complete blocks of length `L_v` are used.

## Synthetic profiles

`fBm_circulant_embedding.py` generates two independent fBm profiles using
circulant embedding. `fBm_with_displacements.py` implements the lateral-
displacement model used in the stability study. `fBm_with_white_noise.py`
implements additive Gaussian height errors. The height-quantisation model is
provided by `Profile_n_bit_conversion.py`, and
`fBm_with_combined_modifications.py` applies displacement, white noise, and
quantisation in the prescribed order.

```python
import numpy as np
from fBm_circulant_embedding import circulant_fBm

rng = np.random.default_rng(1234)
x, z1, z2 = circulant_fBm(H=0.7, L=1000, x_max=999, c_fBm=1.0, rng=rng)
```

A profile with additive white height noise can be generated with:

```python
from fBm_with_white_noise import fBm_with_white_noise

x, z, z_observed = fBm_with_white_noise(
    H=0.7, L=1000, x_max=999, c_fBm=1.0, I=0.0125, rng=rng
)
```

All three modifications can be applied with:

```python
from fBm_with_combined_modifications import fBm_with_combined_modifications

result = fBm_with_combined_modifications(
    H=0.7,
    L=1000,
    x_max=999,
    c_fBm=1.0,
    p0=0.5,
    I=0.0125,
    n_bits=16,
    m=4,
    rng=rng,
)
z_observed = result.z_observed
```

## Reproducing the numerical studies

The full experiments use 10,000 profiles per configuration and may require
substantial runtime and memory. All generators use deterministic random seeds
and record runtime information in their output.

### Parameter studies

Run the method-specific generators:

```bash
python generate_cp_parameter_results.py
python generate_dfa_parameter_results.py
python generate_hfd_parameter_results.py
python generate_psd_parameter_results.py
python generate_sva_parameter_results.py
```

Then aggregate and validate the manuscript parameter choices:

```bash
python analyze_parameter_results.py --input-dir . --output-dir results/parameters
```

This writes a complete configuration summary, the configurations used in the
paper, and a PDF/PNG diagnostic plot. The analysis reports the maximum
absolute estimation error and maximum standard deviation over all simulated
Hurst parameters. It also identifies the Pareto frontier and evaluates the
universal Berry--Esseen bound for `p=0.05` and `epsilon=0.005`. For DFA, the
smallest bound among configurations passing the documented maximum-error
threshold of 0.05 is marked separately. The manuscript choices are validated
explicitly rather than inferred from an undocumented scalar objective.

The full DFA, HFD, and SVA parameter grids are computationally expensive.
For a smoke test, call a generator from Python with fewer profiles and a
configuration limit, for example:

```python
from generate_hfd_parameter_results import generate_hfd_parameter_results

generate_hfd_parameter_results(
    output_path="hfd_smoke_test.csv",
    number_of_profiles=20,
    H_values=(0.2,),
    L_values=(1000,),
    configuration_limit=3,
)
```

### Estimator comparison

```bash
python generate_estimator_comparison_results.py
python plot_estimator_comparison.py
```

The comparison uses the manuscript parameters: CP `(tau=3)`, DFA
`(l=10, f=0.1, n=20)`, HFD and SVA `(l=3, f=0.05, n=10)`, and the fixed
Welch configuration documented in the paper for PSD.

### CP stability tests

```bash
python generate_cp_stability_results.py
python plot_cp_tau_stability.py
```

`generate_cp_stability_results.py` evaluates additive height noise, lateral
sampling-position errors, and finite height-scale bit depth. Missing-data
experiments are kept separate.

The publication figures can be recreated with:

```bash
python CP_Plot.py
python TuningPlot.py
```

`CP_Plot.py` visualises the fixed robustness statistics used in the paper.
`TuningPlot.py` reads `dfa_sva_hfd_parameters_optimised.csv` from the
repository root and creates PDF and PNG versions of the tuning-parameter
comparison.

### Missing-observation table

```bash
python generate_cp_gap_table_data.py
```

The program adaptively generates source profiles until `N_v * L_v` valid
triples are available for each prescribed table configuration. It writes
`cp_gap_table_data.csv` and a JSON metadata file.

## Output files and runtime

CSV files contain numerical results; adjacent `.metadata.txt` or
`.metadata.json` files record parameters, seeds, software versions, and
runtime. Runtime is hardware-dependent and is included to make the scale of
each computation transparent, not as a performance guarantee.

Generated output can be placed in a separate `results/` directory. Large or
machine-specific output files do not need to be committed when they can be
reproduced from the documented scripts.

## Citation

If you use this software, please cite the accompanying paper. The final DOI
and bibliographic details should be added here after publication.

## License

This project is licensed under the MIT License. See `LICENSE`.
