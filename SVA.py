"""Semivariogram analysis (SVA) for one-dimensional profiles.

References
----------
Yang, H.-N., & Lu, T.-M. (1995). Inconsistency between height--height
correlation and power-spectrum functions of scale-invariant surfaces for
roughness exponent alpha < 1. *Physical Review B*, 51, 2479--2483.
https://doi.org/10.1103/PhysRevB.51.2479

Zimmerman, D. L., & Zimmerman, M. B. (1991). A comparison of spatial
semivariogram estimators and corresponding ordinary kriging predictors.
*Technometrics*, 33, 77--91. https://doi.org/10.2307/1269009
"""

import numpy as np


def sva(profile, l_SVA=3, f_SVA=0.05, n_SVA=10, scales=None):
    """
    Semivariogram Analysis (SVA)
    to estimate the Hurst exponent from 1D profiles.

    Parameters
    ----------
    profile : array-like
        Samples ``z_1, ..., z_L`` of an equidistant profile.
    l_SVA : int, default=3
        Minimum fitting scale.
    n_SVA : int, optional
        Number of scales / fitting points (default 10).
    f_SVA : float, optional
        Upper cut-off coefficient for maximum scale τ_n = f_SVA * L (default 0.05).
    scales : array-like, optional
        Explicit integer scales. Otherwise, Eq. (logarithmic-scales) is used.

    Returns
    -------
    H_SVA : float
        Estimated Hurst exponent.
    scales : ndarray of int
        Scales used in the fit.
    M_SVA : ndarray of float
        Semivariogram values from Eq. (sva-scaling).

    Raises
    ------
    ValueError
        If the profile or scale-selection parameters are invalid, if repeated
        scales occur, or if the semivariogram cannot be fitted in log space.

    Notes
    -----
    ``M_SVA`` includes the factor ``1/2`` used in Eq. (sva-scaling). This
    factor changes the fitted intercept but not the Hurst exponent.
    """
    z = np.asarray(profile, dtype=float)
    if z.ndim != 1 or not np.all(np.isfinite(z)):
        raise ValueError("profile must be a one-dimensional array of finite values.")
    L = z.size
    if L < 3:
        raise ValueError("Profile too short for SVA (need at least 3 samples).")

    if scales is None:
        if l_SVA < 1 or n_SVA < 2 or not 0 < f_SVA < 1:
            raise ValueError("Require l_SVA >= 1, n_SVA >= 2, and 0 < f_SVA < 1.")
        raw = l_SVA * ((f_SVA * L) / l_SVA) ** (
            np.arange(n_SVA, dtype=float) / (n_SVA - 1)
        )
        scales = np.floor(raw + 1e-12).astype(int)
    else:
        scales = np.asarray(scales, dtype=int)

    if scales.ndim != 1 or scales.size < 2 or np.any(np.diff(scales) <= 0):
        raise ValueError("scales must contain at least two distinct increasing integers.")
    if scales[0] < 1 or scales[-1] >= L:
        raise ValueError("Each SVA scale must satisfy 1 <= scale < L.")

    # Compute I_τ = (1/(L-τ)) * sum_{k=1}^{L-τ} (y_{k+τ} - y_k)^2
    M_SVA = np.empty(scales.size, dtype=float)
    for i, scale in enumerate(scales):
        differences = z[scale:] - z[:-scale]
        M_SVA[i] = 0.5 * np.mean(differences**2)

    # Log-log linear regression: log I_τ = log C_SVA + 2 H_SVA * log(d τ)
    if np.any(M_SVA <= 0):
        raise ValueError("SVA scaling values must be positive; the profile may be constant.")
    slope = np.polyfit(np.log(scales), np.log(M_SVA), 1)[0]

    H_SVA = 0.5 * slope
    return float(H_SVA), scales, M_SVA
