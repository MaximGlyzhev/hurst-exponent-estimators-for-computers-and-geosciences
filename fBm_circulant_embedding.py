"""Exact fBm simulation on an equidistant grid by circulant embedding.

References
----------
Dietrich, C. R., & Newsam, G. N. (1997). Fast and exact simulation of
stationary Gaussian processes through circulant embedding of the covariance
matrix. *SIAM Journal on Scientific Computing*, 18, 1088--1107.
https://doi.org/10.1137/S1064827592240555
"""

import numpy as np
from typing import Optional, Tuple


def circulant_fBm(
    H: float,
    L: int,
    x_max: float,
    c_fBm: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate two independent fractional Brownian motion (fBm) profiles
    using circulant embedding and the FFT.

    The profiles are evaluated at ``L`` equidistant points on

        [0, x_max].

    The sampling points are

        x_i = i * x_max / (L - 1),

    for i = 0, ..., L - 1.

    The generated fBm satisfies

        Cov(B_H(t), B_H(s))
            = c_fBm / 2
              * (
                    |t|**(2H)
                    + |s|**(2H)
                    - |t-s|**(2H)
                ),

    and therefore

        Var(B_H(t) - B_H(s))
            = c_fBm * |t-s|**(2H).

    Parameters
    ----------
    H : float
        Hurst exponent. Must satisfy 0 < H < 1.

    L : int
        Number of equidistant sampling points, including both endpoints.
        Must be at least 3.

    x_max : float
        Length of the interval on which the fBm is generated. The profile
        is sampled on ``[0, x_max]``. Must be positive.

    c_fBm : float, default=1.0
        Amplitude parameter from the covariance model in the paper:

            Var(B_H(t) - B_H(s))
                = c_fBm * |t-s|**(2H).

    rng : np.random.Generator, optional
        NumPy random-number generator. If None, a new generator is created
        with np.random.default_rng().

    Returns
    -------
    x : np.ndarray, shape (L,)
        Equidistant sampling grid on ``[0, x_max]``.

    fbm1 : np.ndarray, shape (L,)
        First fBm realisation.

    fbm2 : np.ndarray, shape (L,)
        Second independent fBm realisation.

    Notes
    -----
    The spacing between adjacent sampling points is

        dx = x_max / (L - 1).

    Hence, the variance of an increment between adjacent points is

        c_fBm * dx**(2H).

    Changing ``L`` changes the sampling density but not the interval or
    the amplitude parameter of the underlying
    continuous-time fBm.

    Raises
    ------
    TypeError
        If an input has an incompatible type.
    ValueError
        If ``H``, ``L``, ``x_max``, or ``c_fBm`` lies outside its admissible
        range.
    RuntimeError
        If numerical checks indicate an invalid circulant embedding.
    """
    # ------------------------------------------------------------------
    # Validate H
    # ------------------------------------------------------------------
    if not isinstance(H, (int, float, np.integer, np.floating)):
        raise TypeError("H must be a real number.")

    H = float(H)

    if not np.isfinite(H) or not 0.0 < H < 1.0:
        raise ValueError("H must be finite and satisfy 0 < H < 1.")

    # ------------------------------------------------------------------
    # Validate L
    # ------------------------------------------------------------------
    if (
        not isinstance(L, (int, np.integer))
        or isinstance(L, (bool, np.bool_))
    ):
        raise TypeError("L must be an integer.")

    L = int(L)

    if L < 3:
        raise ValueError("L must be at least 3.")

    # ------------------------------------------------------------------
    # Validate x_max
    # ------------------------------------------------------------------
    if not isinstance(
        x_max,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError("x_max must be a real number.")

    x_max = float(x_max)

    if not np.isfinite(x_max) or x_max <= 0.0:
        raise ValueError(
            "x_max must be finite and strictly positive."
        )

    # ------------------------------------------------------------------
    # Validate c_fBm
    # ------------------------------------------------------------------
    if not isinstance(
        c_fBm,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError("c_fBm must be a real number.")

    c_fBm = float(c_fBm)

    if not np.isfinite(c_fBm) or c_fBm <= 0.0:
        raise ValueError("c_fBm must be finite and strictly positive.")

    # ------------------------------------------------------------------
    # Random-number generator
    # ------------------------------------------------------------------
    if rng is None:
        rng = np.random.default_rng()

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------
    number_of_increments = L - 1

    grid_spacing = (
        x_max
        / number_of_increments
    )

    x = np.linspace(
        0.0,
        x_max,
        L,
        dtype=np.float64,
    )

    # ------------------------------------------------------------------
    # Circulant embedding dimensions
    # ------------------------------------------------------------------
    embedding_length = 2 * number_of_increments
    two_H = 2.0 * H

    # ------------------------------------------------------------------
    # Autocovariance of unit-grid fractional Gaussian noise
    #
    # gamma(k) = 1/2 * (
    #     |k+1|^(2H)
    #     + |k-1|^(2H)
    #     - 2|k|^(2H)
    # )
    #
    # At this stage, the covariance corresponds to:
    #
    #     c_fBm = 1
    #     grid spacing = 1
    # ------------------------------------------------------------------
    k = np.arange(
        number_of_increments + 1,
        dtype=np.float64,
    )

    gamma = 0.5 * (
        (k + 1.0) ** two_H
        + np.abs(k - 1.0) ** two_H
        - 2.0 * k**two_H
    )

    # ------------------------------------------------------------------
    # First row of the embedded circulant covariance matrix:
    #
    # gamma(0), gamma(1), ..., gamma(n),
    # gamma(n-1), ..., gamma(1)
    # ------------------------------------------------------------------
    circulant_row = np.concatenate(
        (
            gamma,
            gamma[-2:0:-1],
        )
    )

    # ------------------------------------------------------------------
    # Eigenvalues of the circulant covariance matrix
    #
    # Since the input is real and symmetric, the FFT should be real up
    # to floating-point round-off.
    # ------------------------------------------------------------------
    eigenvalues_complex = np.fft.fft(circulant_row)

    maximum_imaginary_part = float(
        np.max(np.abs(eigenvalues_complex.imag))
    )

    imaginary_tolerance = (
        1e-10
        * max(
            1.0,
            float(np.max(np.abs(eigenvalues_complex.real))),
        )
    )

    if maximum_imaginary_part > imaginary_tolerance:
        raise RuntimeError(
            "Unexpectedly large imaginary components occurred in the "
            "circulant eigenvalues. "
            f"Maximum imaginary part: {maximum_imaginary_part:.6e}."
        )

    eigenvalues = eigenvalues_complex.real

    # ------------------------------------------------------------------
    # Check non-negativity
    #
    # Tiny negative eigenvalues may occur because of floating-point
    # round-off. Substantial negative values indicate that the chosen
    # circulant embedding is not positive semidefinite.
    # ------------------------------------------------------------------
    eigenvalue_scale = max(
        1.0,
        float(np.max(np.abs(eigenvalues))),
    )

    eigenvalue_tolerance = 1e-12 * eigenvalue_scale

    minimum_eigenvalue = float(np.min(eigenvalues))

    if minimum_eigenvalue < -eigenvalue_tolerance:
        raise RuntimeError(
            "The circulant embedding has a significantly negative "
            "eigenvalue. "
            f"Minimum eigenvalue: {minimum_eigenvalue:.6e}."
        )

    # Remove only negligible negative values caused by numerical errors.
    np.maximum(eigenvalues, 0.0, out=eigenvalues)

    # ------------------------------------------------------------------
    # Spectral amplitude
    #
    # NumPy's inverse FFT includes division by embedding_length.
    # Multiplication by sqrt(embedding_length) compensates for this
    # normalization.
    #
    # The real and imaginary parts of the resulting complex sequence
    # produce two independent fGn realisations.
    # ------------------------------------------------------------------
    spectral_amplitude = np.sqrt(
        embedding_length * eigenvalues
    )

    real_noise = rng.standard_normal(embedding_length)
    imaginary_noise = rng.standard_normal(embedding_length)

    complex_white_noise = real_noise + 1j * imaginary_noise

    complex_fgn = np.fft.ifft(
        spectral_amplitude * complex_white_noise
    )

    fgn1 = complex_fgn.real[:number_of_increments]
    fgn2 = complex_fgn.imag[:number_of_increments]

    # ------------------------------------------------------------------
    # Amplitude scaling
    #
    # A unit-grid fBm with c_fBm = 1 satisfies
    #
    #     Var(B_H(i+k) - B_H(i)) = k^(2H).
    #
    # For a grid spacing dx, self-similarity requires multiplication by
    #
    #     dx^H.
    #
    # Multiplication by sqrt(c_fBm) produces the desired amplitude
    # coefficient. Consequently,
    #
    #     Var(B_H(t) - B_H(s))
    #         = c_fBm * |t-s|^(2H).
    # ------------------------------------------------------------------
    amplitude_scale = (
        np.sqrt(c_fBm)
        * grid_spacing**H
    )

    fgn1 *= amplitude_scale
    fgn2 *= amplitude_scale

    # ------------------------------------------------------------------
    # Integrate the fGn sequences to obtain fBm
    # ------------------------------------------------------------------
    fbm1 = np.empty(L, dtype=np.float64)
    fbm2 = np.empty(L, dtype=np.float64)

    fbm1[0] = 0.0
    fbm2[0] = 0.0

    np.cumsum(
        fgn1,
        out=fbm1[1:],
    )

    np.cumsum(
        fgn2,
        out=fbm2[1:],
    )

    return x, fbm1, fbm2
