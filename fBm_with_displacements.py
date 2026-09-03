"""Simulation of the lateral-displacement model defined in the paper.

The discretised truncated-normal displacement distribution and the order of
operations follow Section 2.5 of the main manuscript and Section IV of the
Supplementary Material. fBm profiles are generated using Dietrich and
Newsam's (1997) circulant-embedding method.

References
----------
Dietrich, C. R., & Newsam, G. N. (1997). Fast and exact simulation of
stationary Gaussian processes through circulant embedding of the covariance
matrix. *SIAM Journal on Scientific Computing*, 18, 1088--1107.
https://doi.org/10.1137/S1064827592240555
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf, ndtr
from fBm_circulant_embedding import circulant_fBm


def sigma_from_no_displacement_probability(
    p0: float,
    m: int = 4,
    d: float = 1.0,
) -> float:
    """Return the truncated-normal scale associated with ``p0``.

    Parameters
    ----------
    p0 : float
        Required probability of zero displacement.
    m : int, default=4
        Number of admissible displacement indices on either side.
    d : float, default=1.0
        Nominal distance between adjacent profile points.

    Returns
    -------
    float
        Standard deviation ``sigma`` in the same spatial unit as ``d``.
    """
    if not isinstance(m, (int, np.integer)) or m < 1:
        raise ValueError("m must be a positive integer.")
    if not np.isfinite(d) or d <= 0.0:
        raise ValueError("d must be finite and positive.")
    lower_limit = 1.0 / (2 * m + 1)
    if not np.isfinite(p0) or not lower_limit < p0 < 1.0:
        raise ValueError(f"p0 must satisfy {lower_limit} < p0 < 1.")

    cell_width = d / (2 * m + 1)

    def probability(sigma: float) -> float:
        numerator = erf(cell_width / (2.0 * np.sqrt(2.0) * sigma))
        denominator = erf(d / (2.0 * np.sqrt(2.0) * sigma))
        return float(numerator / denominator)

    return float(
        brentq(
            lambda sigma: probability(sigma) - p0,
            d * 1.0e-8,
            d * 1.0e3,
        )
    )


def generate_displacement_indices(
    sigma: float,
    d: float,
    m: int,
    N: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate N independent realizations of the discrete random variable X.

    For l = -m, ..., m, the probability

        P(X = l) = p_l

    is defined as the probability that a normally distributed random
    variable Z ~ N(0, sigma^2) lies in the l-th subinterval of
    [-d/2, d/2], divided by the probability that Z lies in [-d/2, d/2].

    Parameters
    ----------
    sigma : float
        Standard deviation of the normal distribution.
        Must satisfy sigma > 0.

    d : float
        Length of the conditioning interval.
        Must satisfy d > 0.

    m : int
        Positive integer. The random variable X takes values in
        {-m, ..., m}.

    N : int
        Number of independent realizations to generate.

    rng : numpy.random.Generator, optional
        Random-number generator used for reproducible simulations.

    Returns
    -------
    samples : np.ndarray
        Array of length N containing independent realizations of X.

    Raises
    ------
    ValueError
        If ``sigma``, ``d``, ``m``, or ``N`` is outside its admissible range,
        or if the truncated-normal probabilities cannot be normalised.
    TypeError
        If ``rng`` is not a NumPy random-number generator.
    """

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    if d <= 0:
        raise ValueError("d must be positive.")

    if not isinstance(m, (int, np.integer)) or m < 1:
        raise ValueError("m must be a positive integer.")

    if not isinstance(N, (int, np.integer)) or N < 1:
        raise ValueError("N must be a positive integer.")

    l_values = np.arange(-m, m + 1, dtype=int)

    delta = d / (2 * m + 1)

    lower_bounds = (l_values - 0.5) * delta
    upper_bounds = (l_values + 0.5) * delta

    normalization = (
        ndtr(d / (2 * sigma))
        - ndtr(-d / (2 * sigma))
    )

    if normalization <= 0:
        raise ValueError(
            "The probability of the conditioning interval is numerically "
            "equal to zero. The values of sigma and d may be too differently "
            "scaled."
        )

    probabilities = (
        ndtr(upper_bounds / sigma)
        - ndtr(lower_bounds / sigma)
    ) / normalization

    probabilities = np.maximum(probabilities, 0.0)
    probabilities /= probabilities.sum()

    if rng is None:
        rng = np.random.default_rng()
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")

    samples = rng.choice(
        l_values,
        size=N,
        replace=True,
        p=probabilities
    )

    return samples


def fBm_with_displacements(
    H: float,
    L: int,
    x_max: float,
    c_fBm: float,
    m: int,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the laterally displaced fBm profile defined in the paper.

    Parameters
    ----------
    H, L, c_fBm
        fBm parameters used in the manuscript.
    x_max : float
        Largest nominal profile coordinate; the spacing is
        ``d = x_max/(L-1)``.
    m : int
        Displacement granularity. Indices take values ``-m, ..., m``.
    sigma : float
        Standard deviation of the truncated Gaussian displacement model.
    rng : numpy.random.Generator, optional
        Shared generator for both fBm and displacement sampling.

    Returns
    -------
    x_observed, z_observed : ndarray
        Nominal equidistant coordinates and heights sampled at displaced
        positions, each containing ``L`` values.
    """
    if not isinstance(L, (int, np.integer)) or L < 3:
        raise ValueError("L must be an integer greater than or equal to 3.")
    if not isinstance(m, (int, np.integer)) or m < 1:
        raise ValueError("m must be a positive integer.")
    if rng is None:
        rng = np.random.default_rng()
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")

    d = x_max / (L - 1)
    refinement = 2 * m + 1
    fine_spacing = d / refinement
    fine_L = (L - 1) * refinement + 2 * m + 1
    fine_x_max = (fine_L - 1) * fine_spacing

    _, fine_profile, _ = circulant_fBm(
        H=H,
        L=fine_L,
        x_max=fine_x_max,
        c_fBm=c_fBm,
        rng=rng,
    )
    displacement_indices = generate_displacement_indices(
        sigma=sigma,
        d=d,
        m=m,
        N=L,
        rng=rng,
    )
    nominal_indices = m + refinement * np.arange(L)
    z_observed = fine_profile[nominal_indices + displacement_indices]
    x_observed = np.linspace(0.0, x_max, L)
    return x_observed, z_observed




