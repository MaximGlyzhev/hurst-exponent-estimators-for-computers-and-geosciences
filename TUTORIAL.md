# Tutorial

This tutorial provides a compact, reproducible workflow for generating a
synthetic fracture profile, applying all five Hurst exponent estimators, and
testing the stability extensions used in the paper.

## 1. Set up the environment

```bash
git clone https://github.com/MaximGlyzhev/hurst-exponent-estimators-for-computers-and-geosciences.git
cd hurst-exponent-estimators-for-computers-and-geosciences
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## 2. Generate a reproducible fBm profile

```python
import numpy as np
from fBm_circulant_embedding import circulant_fBm

rng = np.random.default_rng(1234)
x, z, _ = circulant_fBm(
    H=0.5,
    L=1000,
    x_max=999,
    c_fBm=1.0,
    rng=rng,
)

print(x.shape, z.shape)
```

The expected shapes are `(1000,)`. A different seed produces another exact
fBm realisation with the same model parameters.

## 3. Apply all five estimators

```python
from CP import estimate_hurst_cp
from DFA import dfa
from HFD import higuchi_fd
from PSD import psd
from SVA import sva

H_CP = estimate_hurst_cp(z, tau=3)
H_DFA, scales_DFA, M_DFA = dfa(
    z, l_DFA=10, f_DFA=0.1, n_DFA=20
)
D_HFD, H_HFD = higuchi_fd(
    z, l_HFD=3, f_HFD=0.05, n_HFD=10
)
H_PSD, frequencies, M_PSD = psd(z)
H_SVA, scales_SVA, M_SVA = sva(
    z, l_SVA=3, f_SVA=0.05, n_SVA=10
)

print({
    "CP": H_CP,
    "DFA": H_DFA,
    "HFD": H_HFD,
    "PSD": H_PSD,
    "SVA": H_SVA,
})
```

Individual estimates fluctuate around the generating Hurst parameter. The
paper compares averages, standard deviations, and third central moments over
many independent profiles rather than interpreting a single realisation.

## 4. Add white height noise

```python
from fBm_with_white_noise import fBm_with_white_noise

rng_noise = np.random.default_rng(2026)
x_noise, z_true, z_noisy = fBm_with_white_noise(
    H=0.5,
    L=1000,
    x_max=999,
    c_fBm=1.0,
    I=0.0125,
    rng=rng_noise,
)

print(estimate_hurst_cp(z_noisy, tau=3))
```

`I=0.0125` is the representative low-intensity value used in the paper.

## 5. Apply all three measurement modifications

```python
from fBm_with_combined_modifications import fBm_with_combined_modifications

rng_modified = np.random.default_rng(77)
result = fBm_with_combined_modifications(
    H=0.5,
    L=1000,
    x_max=999,
    c_fBm=1.0,
    p0=0.5,
    I=0.0125,
    n_bits=16,
    m=4,
    rng=rng_modified,
)

H_modified = estimate_hurst_cp(result.z_observed, tau=3)
print(H_modified)
```

The modifications are applied in the same order as in the stability study:
lateral sampling-position errors, additive white height noise, and finite
height resolution.

## 6. Estimate H from profiles with missing observations

```python
from CP_gapped import estimate_hurst_cp_gapped

rng_gaps = np.random.default_rng(99)
z_gapped = z.copy()
z_gapped[rng_gaps.random(z.size) < 0.05] = np.nan

result_gapped = estimate_hurst_cp_gapped(
    z_gapped,
    tau=3,
    L_v=100,
    N_v=5,
)

print(result_gapped.H)
print(result_gapped.standard_deviation)
print(result_gapped.number_of_used_triples)
```

Increase `N_v` or `L_v` only when enough valid triples are available. Several
profiles can be supplied as a rectangular array or as an iterable of arrays.

## 7. Run a small parameter-study smoke test

The full study is computationally expensive. Verify the environment first
with a small run:

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

Inspect the CSV and metadata output before starting a full simulation.

## 8. Reproduce the manuscript studies

Run the full data generation only when sufficient runtime and memory are
available:

```bash
python generate_cp_parameter_results.py
python generate_dfa_parameter_results.py
python generate_hfd_parameter_results.py
python generate_psd_parameter_results.py
python generate_sva_parameter_results.py

python analyze_parameter_results.py --input-dir . --output-dir results/parameters

python generate_estimator_comparison_results.py
python plot_estimator_comparison.py

python generate_cp_stability_results.py
python plot_cp_tau_stability.py

python generate_cp_gap_table_data.py
```

The scripts use deterministic seeds and record parameters and runtime in the
output metadata. Exact runtime depends on processor, memory, operating system,
and numerical-library versions.

## 9. Use an external fracture profile

For an external regularly sampled profile stored as one height value per line:

```python
import numpy as np
from CP import estimate_hurst_cp

z_measured = np.loadtxt("profile.csv")
H_measured = estimate_hurst_cp(z_measured, tau=3)
print(H_measured)
```

Detrend the profile and verify that the sampling grid is regular before using
the estimators. If the file contains missing observations, encode them as
`numpy.nan` and use `estimate_hurst_cp_gapped` instead.

For complete parameter definitions and error conditions, see
[`USER_GUIDE.md`](USER_GUIDE.md).
