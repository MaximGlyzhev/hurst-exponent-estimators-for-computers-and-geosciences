"""Generate fractional Brownian motion profiles with additive white noise.

The observation model follows the accompanying paper:

    z_i_obs = z_i + epsilon_i,  epsilon_i ~ N(0, I**2),

where the errors are independent and ``I`` is the height-noise intensity.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from fBm_circulant_embedding import circulant_fBm


def add_white_noise(
    profile: ArrayLike,
    I: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add independent centred Gaussian height errors to a profile.

    Parameters
    ----------
    profile : array-like
        One-dimensional profile containing the unmodified heights.
    I : float
        Standard deviation of the additive white noise. It must be finite and
        non-negative and use the same height unit as ``profile``.
    rng : numpy.random.Generator, optional
        Random-number generator for reproducible simulations.

    Returns
    -------
    ndarray
        A noisy floating-point copy of the profile.
    """
    z = np.asarray(profile, dtype=float)
    if z.ndim != 1 or z.size == 0 or not np.all(np.isfinite(z)):
        raise ValueError("profile must be a non-empty 1D array of finite values.")
    if not np.isfinite(I) or I < 0.0:
        raise ValueError("I must be finite and non-negative.")
    if rng is None:
        rng = np.random.default_rng()
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")
    return z + rng.normal(loc=0.0, scale=float(I), size=z.size)


def fBm_with_white_noise(
    H: float,
    L: int,
    x_max: float,
    c_fBm: float,
    I: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one fBm profile and add independent white height noise.

    Parameters
    ----------
    H, L, x_max, c_fBm
        fBm parameters in the notation of the accompanying paper.
    I : float
        Standard deviation of the additive height error.
    rng : numpy.random.Generator, optional
        Shared random-number generator used for fBm and noise simulation.

    Returns
    -------
    x, z, z_observed : ndarray
        Regular grid, unmodified fBm heights, and noisy observed heights.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")
    x, z, _ = circulant_fBm(
        H=H, L=L, x_max=x_max, c_fBm=c_fBm, rng=rng
    )
    return x, z, add_white_noise(z, I=I, rng=rng)
