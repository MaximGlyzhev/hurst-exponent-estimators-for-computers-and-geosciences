"""Modified change-probability estimator for regularly sampled gapped data.

Missing observations may be represented by ``numpy.nan`` values or supplied
through an explicit Boolean observation mask. Valid CP triples are ordered by
profile and starting index, then divided into ``N_v`` evaluation blocks of
``L_v`` triples, following the construction in the accompanying paper.

References
----------
Gutjahr, T., Hale, S., Keller, K., Blum, P., & Winter, S. (2022).
Quantification of fracture roughness by change probabilities and Hurst
exponents. Mathematical Geosciences, 54, 679--710.
https://doi.org/10.1007/s11004-021-09985-3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike

from CP import hurst_from_change_probability


@dataclass(frozen=True)
class GappedCPResult:
    """Results returned by :func:`estimate_hurst_cp_gapped`.

    Attributes
    ----------
    H : float
        Mean of the ``N_v`` block-wise Hurst exponent estimates.
    standard_deviation : float
        Sample standard deviation of the block-wise estimates. It is ``nan``
        when ``N_v=1`` because a sample standard deviation is then undefined.
    H_blocks : ndarray, shape (N_v,)
        Hurst exponent estimate for every evaluation block.
    p_blocks : ndarray, shape (N_v,)
        Empirical change probability for every evaluation block.
    tau, L_v, N_v : int
        Scale parameter, valid triples per block, and number of blocks.
    number_of_valid_triples : int
        Total number of valid triples found in all supplied profiles.
    number_of_used_triples : int
        Number ``N_v * L_v`` of triples used by the estimator.
    number_of_discarded_triples : int
        Valid triples remaining after the complete blocks were formed.
    """

    H: float
    standard_deviation: float
    H_blocks: np.ndarray
    p_blocks: np.ndarray
    tau: int
    L_v: int
    N_v: int
    number_of_valid_triples: int
    number_of_used_triples: int
    number_of_discarded_triples: int


def _profiles_as_list(profiles: ArrayLike | Iterable[ArrayLike]) -> list[np.ndarray]:
    """Normalise one profile, a 2D array, or an iterable of profiles."""
    try:
        array = np.asarray(profiles, dtype=float)
    except (TypeError, ValueError):
        array = None
    if array is not None:
        if array.ndim == 1:
            return [array]
        if array.ndim == 2:
            return [row for row in array]
        if array.ndim != 0:
            raise ValueError("profiles must be a 1D or 2D array.")

    try:
        result = [np.asarray(profile, dtype=float) for profile in profiles]
    except TypeError as error:
        raise TypeError("profiles must be an array or an iterable of arrays.") from error
    if not result:
        raise ValueError("At least one profile must be supplied.")
    if any(profile.ndim != 1 for profile in result):
        raise ValueError("Every profile must be one-dimensional.")
    return result


def valid_change_indicators(
    profile: ArrayLike,
    tau: int = 3,
    observed: ArrayLike | None = None,
) -> np.ndarray:
    """Return change indicators for all valid triples in one gapped profile.

    Parameters
    ----------
    profile : array-like
        Heights on a regular sampling grid. Missing values can be represented
        by ``numpy.nan``.
    tau : int, default=3
        Positive CP scale parameter, expressed in grid intervals.
    observed : array-like of bool, optional
        Explicit observation mask. ``True`` denotes an available observation.
        Non-finite profile values are treated as missing even when the mask is
        ``True``.

    Returns
    -------
    ndarray of uint8
        Change indicators for valid triples, ordered by starting index.
    """
    if not isinstance(tau, (int, np.integer)) or isinstance(
        tau, (bool, np.bool_)
    ):
        raise TypeError("tau must be an integer.")
    if tau < 1:
        raise ValueError("tau must be positive.")

    z = np.asarray(profile, dtype=float)
    if z.ndim != 1:
        raise ValueError("profile must be one-dimensional.")
    if z.size <= 2 * tau:
        return np.empty(0, dtype=np.uint8)

    finite = np.isfinite(z)
    if observed is None:
        available = finite
    else:
        available = np.asarray(observed, dtype=bool)
        if available.shape != z.shape:
            raise ValueError("observed must have the same shape as profile.")
        available = available & finite

    number_of_triples = z.size - 2 * tau
    valid = (
        available[:number_of_triples]
        & available[tau : tau + number_of_triples]
        & available[2 * tau :]
    )
    a = z[:number_of_triples]
    b = z[tau : tau + number_of_triples]
    c = z[2 * tau :]
    changes = ((a < b) & (b >= c)) | ((a >= b) & (b < c))
    return changes[valid].astype(np.uint8, copy=False)


def estimate_hurst_cp_gapped(
    profiles: ArrayLike | Iterable[ArrayLike],
    *,
    tau: int = 3,
    L_v: int = 1000,
    N_v: int | None = None,
    observed_masks: ArrayLike | Iterable[ArrayLike] | None = None,
) -> GappedCPResult:
    """Estimate a Hurst exponent from regularly sampled gapped profiles.

    Parameters
    ----------
    profiles : array-like or iterable of array-like
        One profile, a rectangular array with one profile per row, or an
        iterable of possibly different-length profiles. Missing observations
        may be encoded as ``numpy.nan``.
    tau : int, default=3
        CP scale parameter in grid intervals.
    L_v : int, default=1000
        Number of valid triples in each evaluation block.
    N_v : int, optional
        Number of evaluation blocks. If omitted, all complete blocks of size
        ``L_v`` are used.
    observed_masks : array-like or iterable of array-like, optional
        Boolean masks corresponding to ``profiles``. ``True`` marks an
        observed value. Omit this argument when gaps are encoded as ``nan``.

    Returns
    -------
    GappedCPResult
        Mean estimate, sample standard deviation, block-wise estimates and
        change probabilities, and triple counts.

    Raises
    ------
    ValueError
        If a parameter is invalid or fewer than ``N_v * L_v`` valid triples
        are available.

    Examples
    --------
    >>> z = np.array([0.0, 1.0, np.nan, 0.5, 0.7, 0.9, 1.1])
    >>> result = estimate_hurst_cp_gapped(z, tau=1, L_v=2, N_v=1)
    >>> result.N_v
    1
    """
    if not isinstance(L_v, (int, np.integer)) or isinstance(L_v, (bool, np.bool_)):
        raise TypeError("L_v must be an integer.")
    if L_v < 1:
        raise ValueError("L_v must be positive.")
    if N_v is not None:
        if not isinstance(N_v, (int, np.integer)) or isinstance(
            N_v, (bool, np.bool_)
        ):
            raise TypeError("N_v must be an integer or None.")
        if N_v < 1:
            raise ValueError("N_v must be positive.")

    profile_list = _profiles_as_list(profiles)
    if observed_masks is None:
        mask_list: list[np.ndarray | None] = [None] * len(profile_list)
    else:
        masks = _profiles_as_list(observed_masks)
        if len(masks) != len(profile_list):
            raise ValueError("One observed mask is required for every profile.")
        mask_list = masks

    indicator_arrays = [
        valid_change_indicators(profile, tau=tau, observed=mask)
        for profile, mask in zip(profile_list, mask_list)
    ]
    indicators = (
        np.concatenate(indicator_arrays)
        if indicator_arrays
        else np.empty(0, dtype=np.uint8)
    )
    number_of_valid_triples = int(indicators.size)

    if N_v is None:
        N_v = number_of_valid_triples // L_v
        if N_v == 0:
            raise ValueError(
                f"Only {number_of_valid_triples} valid triples are available; "
                f"at least L_v={L_v} are required."
            )
    number_of_used_triples = N_v * L_v
    if number_of_valid_triples < number_of_used_triples:
        raise ValueError(
            f"Only {number_of_valid_triples} valid triples are available, but "
            f"N_v*L_v={number_of_used_triples} are required."
        )

    blocks = indicators[:number_of_used_triples].reshape(N_v, L_v)
    p_blocks = np.mean(blocks, axis=1)
    H_blocks = np.array(
        [hurst_from_change_probability(float(p)) for p in p_blocks], dtype=float
    )
    if not np.all(np.isfinite(H_blocks)):
        raise ValueError(
            "At least one block produced a non-finite Hurst estimate. "
            "Increase L_v or inspect the data for degenerate ordinal patterns."
        )

    H = float(np.mean(H_blocks))
    standard_deviation = (
        float(np.std(H_blocks, ddof=1)) if N_v >= 2 else float("nan")
    )
    return GappedCPResult(
        H=H,
        standard_deviation=standard_deviation,
        H_blocks=H_blocks,
        p_blocks=p_blocks,
        tau=tau,
        L_v=L_v,
        N_v=N_v,
        number_of_valid_triples=number_of_valid_triples,
        number_of_used_triples=number_of_used_triples,
        number_of_discarded_triples=(
            number_of_valid_triples - number_of_used_triples
        ),
    )
