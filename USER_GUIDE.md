# User guide

This guide documents the public interfaces of the code accompanying
*A comparison of Hurst exponent estimators for fracture roughness analysis*.
The notation follows the manuscript wherever possible.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Input conventions

- A profile is a one-dimensional array of height values `z` with length `L`.
- Profiles are assumed to be sampled on a regular grid unless stated otherwise.
- The estimator modules require finite values. Use `CP_gapped.py` when
  observations are missing.
- The manuscript parameters are Hurst exponent `H`, profile length `L`, CP
  scale `tau`, minimum fitting scale `l`, upper cut-off coefficient `f`, and
  number of fitting scales `n`.
- All random generators accept a NumPy random generator through `rng`. Supply
  a fixed seed for reproducible output.

## Hurst exponent estimators

### Change probability (`CP.py`)

```python
estimate_change_probability(profile, tau=3) -> float
hurst_from_change_probability(change_probability) -> float
estimate_hurst_cp(profile, tau=3) -> float
```

`tau` is measured in grid intervals. `estimate_hurst_cp` returns the CP Hurst
exponent estimate. At least `2*tau + 1` profile points are required.

### CP for missing observations (`CP_gapped.py`)

```python
estimate_hurst_cp_gapped(
    profiles,
    tau=3,
    L_v=1000,
    N_v=None,
    observed_masks=None,
) -> GappedCPResult
```

Missing values may be encoded as `numpy.nan` or specified by Boolean
`observed_masks`. `L_v` is the number of valid triples per virtual evaluation
block and `N_v` is the requested number of blocks. If `N_v` is omitted, all
complete blocks are used. The result contains the mean Hurst estimate, sample
standard deviation, block-wise estimates and change probabilities, and the
numbers of available and used triples.

### Detrended fluctuation analysis (`DFA.py`)

```python
dfa(profile, order=1, l_DFA=10, f_DFA=0.1, n_DFA=20, scales=None)
```

Returns `(H_DFA, scales, M_DFA)`. Explicit integer `scales` override
`l_DFA`, `f_DFA`, and `n_DFA`. The integration step commonly used for
fractional Gaussian noise is omitted because the input profiles are modelled
as fractional Brownian motion.

### Higuchi fractal dimension (`HFD.py`)

```python
higuchi_fd(profile, l_HFD=3, f_HFD=0.05, n_HFD=10)
```

Returns `(D, H)`, where `D` is the Higuchi fractal dimension and `H = 2-D`.
The fitting scales follow the logarithmic-scale rule in the manuscript.

### Power spectral density (`PSD.py`)

```python
psd(
    profile,
    l_Welch=None,
    O_Welch=None,
    F_Welch=None,
    fs=1.0,
    window="hann",
    detrend="constant",
    average="median",
    scaling="density",
    nu_min=None,
    nu_max=0.05,
)
```

Returns `(H_PSD, frequencies, M_PSD)`. Defaults are
`l_Welch=floor(L/2)`, `O_Welch=floor(L/4)`, `F_Welch=L`, and
`nu_min=4/L`. The fit uses the positive frequencies between `nu_min` and
`nu_max`.

### Semivariogram analysis (`SVA.py`)

```python
sva(profile, l_SVA=3, f_SVA=0.05, n_SVA=10, scales=None)
```

Returns `(H_SVA, scales, M_SVA)`. Explicit integer `scales` override the
three scale-selection parameters.

## Synthetic profiles and measurement modifications

### Exact one-dimensional fBm profiles

```python
circulant_fBm(H, L, x_max, c_fBm=1.0, rng=None)
```

`fBm_circulant_embedding.py` returns a regular coordinate vector and two
independent fBm profiles. `c_fBm` controls amplitude, not the target Hurst
parameter.

### White height noise

`fBm_with_white_noise.py` provides `add_white_noise` for an existing profile
and `fBm_with_white_noise` for generation plus modification. `I` is the
noise-to-amplitude parameter used in the manuscript.

### Lateral sampling-position errors

`fBm_with_displacements.py` implements the truncated, discretised Gaussian
displacement model. `p0` is the probability of no displacement and `m` is the
number of possible displacement positions on either side of the nominal grid
point.

### Finite height resolution

```python
quantize_profile_nbit(profile, n_bits)
```

`Profile_n_bit_conversion.py` maps the profile to the requested bit depth.

### Combined modifications

`fBm_with_combined_modifications.py` applies lateral displacement, white
height noise, and height quantisation in the order used in the paper. Its
result object contains the unmodified and intermediate profiles as well as
the final observed profile.

## Reproducing the numerical studies

The full experiments use 10,000 profiles per configuration and can require
substantial CPU time and memory. The generation scripts write CSV result
tables and adjacent metadata files containing parameters, seeds, software
versions, and runtime.

| Study | Data generation | Analysis or plot |
|---|---|---|
| CP parameters | `generate_cp_parameter_results.py` | `analyze_parameter_results.py` |
| DFA parameters | `generate_dfa_parameter_results.py` | `analyze_parameter_results.py` |
| HFD parameters | `generate_hfd_parameter_results.py` | `analyze_parameter_results.py` |
| PSD parameters | `generate_psd_parameter_results.py` | `analyze_parameter_results.py` |
| SVA parameters | `generate_sva_parameter_results.py` | `analyze_parameter_results.py` |
| Estimator comparison | `generate_estimator_comparison_results.py` | `plot_estimator_comparison.py` |
| CP stability | `generate_cp_stability_results.py` | `plot_cp_tau_stability.py`, `CP_Plot.py` |
| Missing observations | `generate_cp_gap_table_data.py` | table CSV and metadata |

The method-specific parameter generators accept smaller profile counts and
configuration limits when called from Python. Use these options for smoke
tests before starting the full experiment.

## Expected behaviour and errors

- Constant profiles cannot be fitted in logarithmic space and raise an error.
- Repeated integer fitting scales raise an error; reduce `n` or enlarge the
  interval from `l` to `f*L`.
- CP requires more than `2*tau` profile points.
- Gapped CP raises an error when fewer than `N_v*L_v` valid triples are
  available.
- PSD requires at least two positive spectral estimates in the selected
  frequency interval.
- Runtime values are descriptive and hardware-dependent; they are not
  performance guarantees.

## Citation and license

Please cite the accompanying paper when using this code. The software is
distributed under the MIT License; see `LICENSE`.
