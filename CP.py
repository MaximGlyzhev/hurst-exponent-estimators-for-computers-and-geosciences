"""Change-probability (CP) estimator used in the accompanying paper.

References
----------
Gutjahr, T., Hale, S., Keller, K., Blum, P., & Winter, S. (2022).
Quantification of fracture roughness by change probabilities and Hurst
exponents. *Mathematical Geosciences*, 54, 679--710.
https://doi.org/10.1007/s11004-021-09985-3

Sinn, M., & Keller, K. (2011). Estimation of ordinal-pattern probabilities
in Gaussian processes with stationary increments. *Computational Statistics
& Data Analysis*, 55, 1781--1790.
https://doi.org/10.1016/j.csda.2010.11.009
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def estimate_change_probability(profile: ArrayLike, tau: int = 3) -> float:
    """
    Estimate the probability of a direction change using lagged triples.

    For every start index i such that i + 2*tau < L, consider the triple:
        a = z[i]
        b = z[i + tau]
        c = z[i + 2*tau]

    A "change" occurs if b is a local extremum relative to a and c:
        - local maximum: a < b and b >= c
        - local minimum: a >= b and b < c

    The estimate is:
        (#changes) / (#triples)

    Parameters
    ----------
    profile : array-like
        One-dimensional profile ``(z_i)`` containing ``L`` heights.
    tau : int, default=3
        Positive CP scale parameter from Eq. (change-probability) of the paper.

    Returns
    -------
    float
        Estimated change probability in [0, 1].

    Raises
    ------
    ValueError
        If ``tau`` is invalid or the profile contains fewer than ``2*tau + 1``
        points.
    """
    if not isinstance(tau, (int, np.integer)) or isinstance(tau, (bool, np.bool_)):
        raise TypeError("tau must be an integer.")
    if tau < 1:
        raise ValueError("tau must be a positive integer.")

    z = np.asarray(profile, dtype=float)
    if z.ndim != 1:
        raise ValueError("profile must be one-dimensional.")
    if not np.all(np.isfinite(z)):
        raise ValueError("profile must contain only finite values.")

    number_of_triples = z.size - 2 * tau
    if number_of_triples <= 0:
        raise ValueError("profile must contain more than 2*tau points.")

    a = z[:number_of_triples]
    b = z[tau:tau + number_of_triples]
    c = z[2 * tau:]

    maxima = (a < b) & (b >= c)
    minima = (a >= b) & (b < c)

    changes = np.count_nonzero(maxima | minima)
    return float(changes / number_of_triples)


def hurst_from_change_probability(change_probability: float) -> float:
    """
    Estimate the Hurst exponent H (for an fBm model) from a change probability p.

        theta = pi * (1 - p) / 2
        H = 1 + log2(sin(theta))

    Parameters
    ----------
    change_probability
        Change probability p in [0, 1].

    Returns
    -------
    float
        Estimated Hurst exponent.

    Raises
    ------
    ValueError
        If change_probability is not in [0, 1].
    """
    p = float(change_probability)
    if not (0.0 <= p <= 1.0):
        raise ValueError("change_probability must be in [0, 1]")

    if p == 1.0:
        return -np.inf

    # sin(pi(1-p)/2) = cos(pi*p/2), as in Eq. (cp-estimator).
    with np.errstate(divide="ignore"):
        return float(1.0 + np.log2(np.cos(np.pi * p / 2.0)))


def estimate_hurst_cp(profile: ArrayLike, tau: int = 3) -> float:
    """Estimate ``H_CP(tau)`` directly from a profile.

    Parameters
    ----------
    profile : array-like
        One-dimensional profile ``(z_i)`` containing ``L`` heights.
    tau : int, default=3
        Positive CP scale parameter.

    Returns
    -------
    float
        Hurst exponent estimate from Eq. (cp-estimator) of the paper.

    Raises
    ------
    TypeError
        If ``tau`` is not an integer.
    ValueError
        If the profile or ``tau`` is invalid.
    """
    p_CP = estimate_change_probability(profile, tau=tau)
    return hurst_from_change_probability(p_CP)
