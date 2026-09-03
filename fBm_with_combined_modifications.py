"""Generate fBm profiles with all three measurement modifications.

The modifications are applied in the order specified in the accompanying
paper: lateral sampling-position errors, additive white height noise, and
finite-bit height quantisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Profile_n_bit_conversion import quantize_profile_nbit
from fBm_with_displacements import (
    fBm_with_displacements,
    sigma_from_no_displacement_probability,
)
from fBm_with_white_noise import add_white_noise


@dataclass(frozen=True)
class CombinedModificationResult:
    """Intermediate and final profiles from the combined modification model."""

    x: np.ndarray
    z_displaced: np.ndarray
    z_displaced_and_noisy: np.ndarray
    z_observed: np.ndarray
    sigma: float
    original_height_range: tuple[float, float]


def fBm_with_combined_modifications(
    H: float,
    L: int,
    x_max: float,
    c_fBm: float,
    p0: float,
    I: float,
    n_bits: int,
    m: int = 4,
    rng: np.random.Generator | None = None,
) -> CombinedModificationResult:
    """Generate an fBm profile affected by all three modifications.

    Parameters
    ----------
    H, L, x_max, c_fBm
        fBm parameters in the notation of the paper.
    p0 : float
        Probability of zero lateral displacement. For a given ``m``, it must
        lie strictly between ``1/(2*m+1)`` and one.
    I : float
        Standard deviation of the additive white height noise.
    n_bits : int
        Bit depth used for the final height quantisation.
    m : int, default=4
        Number of possible displacement indices on either side of a nominal
        sampling position.
    rng : numpy.random.Generator, optional
        Shared random-number generator used by all stochastic operations.

    Returns
    -------
    CombinedModificationResult
        Regular nominal coordinates, intermediate height profiles, final
        observed profile, the derived displacement ``sigma``, and the height
        range used by the quantisation step.

    Notes
    -----
    The operation order is

    1. lateral displacement;
    2. additive white noise;
    3. n-bit height quantisation.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")
    if not isinstance(L, (int, np.integer)) or L < 3:
        raise ValueError("L must be an integer greater than or equal to 3.")

    d = float(x_max) / (L - 1)
    sigma = sigma_from_no_displacement_probability(p0=p0, m=m, d=d)
    x, z_displaced = fBm_with_displacements(
        H=H,
        L=L,
        x_max=x_max,
        c_fBm=c_fBm,
        m=m,
        sigma=sigma,
        rng=rng,
    )
    z_noisy = add_white_noise(z_displaced, I=I, rng=rng)
    z_observed, height_range = quantize_profile_nbit(z_noisy, n_bits=n_bits)
    return CombinedModificationResult(
        x=x,
        z_displaced=z_displaced,
        z_displaced_and_noisy=z_noisy,
        z_observed=z_observed,
        sigma=sigma,
        original_height_range=height_range,
    )
