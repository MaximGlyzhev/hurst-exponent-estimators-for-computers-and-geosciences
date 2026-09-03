"""Higuchi fractal-dimension (HFD) estimator used in the paper.

References
----------
Higuchi, T. (1990). Relationship between the fractal dimension and the
power-law index for a time series: A numerical investigation.
*Physica D: Nonlinear Phenomena*, 46, 254--264.
https://doi.org/10.1016/0167-2789(90)90039-R
"""

import numpy as np

def higuchi_fd(profile, l_HFD=3, f_HFD=0.05, n_HFD=10):
    """
    Estimate the Higuchi fractal dimension and the corresponding
    Hurst exponent for a one-dimensional profile.

    Parameters
    ----------
    profile : array-like
        One-dimensional profile of length L.
    l_HFD : int, default=3
        Minimum fitting scale.
    f_HFD : float, default=0.05
        Upper-scale cut-off coefficient. The largest fitting scale
        is floor(f_HFD * L).
    n_HFD : int, default=10
        Number of approximately logarithmically equidistant fitting
        scales.

    Returns
    -------
    D : float
        Estimated Higuchi fractal dimension.
    H : float
        Estimated Hurst exponent, calculated as H = 2 - D.

    Raises
    ------
    ValueError
        If the profile or scale-selection parameters are invalid, if repeated
        scales occur after rounding, or if the profile produces non-positive
        Higuchi curve lengths.

    Notes
    -----
    The implementation uses the ``s_k^-3`` normalization from Eq.
    (hfd-scaling) of the accompanying paper.
    """
    X = np.asarray(profile, dtype=float)

    if X.ndim != 1:
        raise ValueError("profile must be one-dimensional.")

    L = X.size

    if L < 4:
        raise ValueError("profile is too short.")

    l_HFD = int(l_HFD)
    n_HFD = int(n_HFD)
    f_HFD = float(f_HFD)

    if l_HFD < 1:
        raise ValueError("l_HFD must be a positive integer.")

    if n_HFD < 2:
        raise ValueError("n_HFD must be at least 2.")

    if not 0 < f_HFD <= 0.5:
        raise ValueError("f_HFD must satisfy 0 < f_HFD <= 0.5.")

    if f_HFD * L < l_HFD:
        raise ValueError(
            f"Invalid scale range: f_HFD*L={f_HFD * L:g} is smaller "
            f"than l_HFD={l_HFD}."
        )

    # Select the scales exactly as described in the manuscript:
    #
    # s_j = floor(
    #     l * (f * L / l)^((j - 1) / (n - 1))
    # ),
    # j = 1, ..., n.
    exponents = np.arange(n_HFD, dtype=float) / (n_HFD - 1)

    unrounded_scales = (
        l_HFD
        * ((f_HFD * L) / l_HFD) ** exponents
    )

    # The small tolerance prevents a value that is mathematically an
    # integer, such as 50, from being represented as 49.99999999999999.
    k_arr = np.floor(unrounded_scales + 1e-12).astype(int)

    # Integer rounding can produce repeated scales when the interval
    # is too small for the requested number of fitting points.
    if np.any(np.diff(k_arr) <= 0):
        raise ValueError(
            "The selected values of l_HFD, f_HFD, L, and n_HFD do not "
            "produce distinct fitting scales. Reduce n_HFD or enlarge "
            "the fitting-scale interval."
        )

    M_hfd = np.zeros(k_arr.size, dtype=float)

    for j, k in enumerate(k_arr):
        total_length = 0.0

        # Python index m=0,...,k-1 corresponds to the manuscript
        # index m=1,...,k.
        for m in range(k):
            T_mk = (L - m - 1) // k

            if T_mk < 1:
                continue

            indices = m + k * np.arange(T_mk + 1)

            differences = np.abs(
                np.diff(X[indices])
            )

            total_length += (
                np.sum(differences) / T_mk
            )

        # Original Higuchi normalization:
        #
        # M_HFD(k) = (L - 1) / k^3
        #            * sum_m [
        #                1 / T(m,k)
        #                * sum_i |X_(m+ik)-X_(m+(i-1)k)|
        #              ]
        M_hfd[j] = (
            (L - 1)
            / k**3
            * total_length
        )

    if np.any(M_hfd <= 0):
        raise ValueError(
            "Non-positive Higuchi curve lengths encountered; "
            "the profile may be constant."
        )

    # M_HFD(k) is proportional to k^(H-2) = k^(-D).
    slope = np.polyfit(
        np.log(k_arr.astype(float)),
        np.log(M_hfd),
        1,
    )[0]

    D = -float(slope)
    H = 2.0 - D

    return D, H



