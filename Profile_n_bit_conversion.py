"""Quantisation of profile heights to the n-bit model used in the paper."""

import numpy as np
from typing import Tuple

def quantize_profile_nbit(
    profile: np.ndarray,
    n_bits: int,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Quantize a height profile into n-bit resolution and return
    the discretised profile together with the original height range.

    Parameters
    ----------
    profile : np.ndarray
        Input height profile (float).
    n_bits : int
        Bit depth (e.g. 8, 16, 32).

    Returns
    -------
    q_profile : np.ndarray
        Quantised profile mapped back to float heights.
    (h_min, h_max) : tuple
        Original height range of the input profile.

    Raises
    ------
    ValueError
        If ``profile`` is not one-dimensional, contains non-finite values, or
        if ``n_bits`` is not positive.

    Notes
    -----
    Minimum and maximum heights are preserved, while intermediate values are
    mapped to one of ``2**n_bits`` equally spaced levels, as described in
    Section II.3 of the Supplementary Material.
    """
    y = np.asarray(profile, dtype=float)
    if y.ndim != 1 or y.size == 0 or not np.all(np.isfinite(y)):
        raise ValueError("profile must be a non-empty 1D array of finite values.")

    if not isinstance(n_bits, (int, np.integer)) or isinstance(
        n_bits, (bool, np.bool_)
    ):
        raise ValueError("n_bits must be an integer.")
    if n_bits < 1:
        raise ValueError("n_bits must be >= 1.")

    levels = 2 ** n_bits
    h_min = float(np.min(y))
    h_max = float(np.max(y))

    # constant profile case
    if h_max == h_min:
        return np.full_like(y, h_min), (h_min, h_max)

    # map to integer levels
    scale = (levels - 1) / (h_max - h_min)
    codes = np.rint((y - h_min) * scale)
    codes = np.clip(codes, 0, levels - 1)

    # map back to float heights
    q_profile = h_min + (codes / (levels - 1)) * (h_max - h_min)

    return q_profile, (h_min, h_max)
