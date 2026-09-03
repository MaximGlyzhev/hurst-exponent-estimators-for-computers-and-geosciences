"""Detrended fluctuation analysis (DFA) for fBm-type profiles.

References
----------
Peng, C.-K., Buldyrev, S. V., Havlin, S., Simons, M., Stanley, H. E., &
Goldberger, A. L. (1994). Mosaic organization of DNA nucleotides.
*Physical Review E*, 49, 1685--1689.
https://doi.org/10.1103/PhysRevE.49.1685

Bardet, J.-M., & Kammoun, I. (2008). Asymptotic properties of the detrended
fluctuation analysis of long-range-dependent processes. *IEEE Transactions
on Information Theory*, 54, 2041--2052.
https://doi.org/10.1109/TIT.2008.920328
"""

import numpy as np


def dfa(profile, order=1, l_DFA=10, f_DFA=0.1, n_DFA=20, scales=None):
    """
    Perform Detrended Fluctuation Analysis (DFA) to estimate the Hurst exponent.

    The integration step used when DFA is applied to fGn is omitted because
    fracture profiles are modelled directly as fBm.

    Parameters
    ----------
    profile : array-like
        One-dimensional profile ``(z_i)`` of length ``L``.
    order : int, default=1
        Polynomial detrending order.
    l_DFA, f_DFA, n_DFA
        Minimum scale, upper cut-off coefficient, and number of fitting scales
        from Eq. (logarithmic-scales) of the paper.
    scales : array-like, optional
        Explicit integer scales. If supplied, the three scale-selection
        parameters are ignored.

    Returns
    -------
    H_DFA : float
        Estimated Hurst exponent.
    scales : ndarray
        Scales used in the regression.
    M_DFA : ndarray
        DFA scaling quantity at each scale.

    Raises
    ------
    ValueError
        If the profile, detrending order, or fitting scales are invalid, or
        if a logarithmic regression cannot be performed.

    Notes
    -----
    The returned scaling quantity is Eq. (dfa-scaling) of the paper. Scales
    are selected using Eq. (logarithmic-scales) unless supplied explicitly.
    """
    z = np.asarray(profile, dtype=float)
    if z.ndim != 1 or not np.all(np.isfinite(z)):
        raise ValueError("profile must be a one-dimensional array of finite values.")
    L = z.size
    if not isinstance(order, (int, np.integer)) or order < 0:
        raise ValueError("order must be a non-negative integer.")

    if scales is None:
        if l_DFA <= order or n_DFA < 2 or not 0 < f_DFA <= 1:
            raise ValueError("Require l_DFA > order, n_DFA >= 2, and 0 < f_DFA <= 1.")
        raw = l_DFA * ((f_DFA * L) / l_DFA) ** (
            np.arange(n_DFA, dtype=float) / (n_DFA - 1)
        )
        scales = np.floor(raw + 1e-12).astype(int)
    else:
        scales = np.asarray(scales, dtype=int)

    if scales.ndim != 1 or scales.size < 2 or np.any(np.diff(scales) <= 0):
        raise ValueError("scales must contain at least two distinct increasing integers.")
    if scales[0] <= order or scales[-1] > L:
        raise ValueError("Each scale must satisfy order < scale <= L.")

    M_DFA = []
    for scale in scales:
        number_of_subprofiles = L // scale
        rmsr = []
        for m in range(number_of_subprofiles):
            indices = np.arange(m * scale, (m + 1) * scale)
            coeffs = np.polyfit(indices, z[indices], order)
            trend = np.polyval(coeffs, indices)
            rmsr.append(np.sqrt(np.mean((z[indices] - trend) ** 2)))
        M_DFA.append(np.mean(rmsr))

    M_DFA = np.asarray(M_DFA)
    if np.any(M_DFA <= 0):
        raise ValueError("DFA scaling values must be positive; the profile may be constant.")
    H_DFA = float(np.polyfit(np.log(scales), np.log(M_DFA), 1)[0])

    return H_DFA, scales, M_DFA

