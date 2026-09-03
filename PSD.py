"""Power-spectral-density estimator based on Welch's method.

References
----------
Flandrin, P. (1989). On the spectrum of fractional Brownian motions.
*IEEE Transactions on Information Theory*, 35, 197--199.
https://doi.org/10.1109/18.42195

Welch, P. D. (1967). The use of fast Fourier transform for the estimation of
power spectra: A method based on time averaging over short, modified
periodograms. *IEEE Transactions on Audio and Electroacoustics*, 15, 70--73.
https://doi.org/10.1109/TAU.1967.1161901
"""

import numpy as np
from scipy.signal import welch


def psd(
    profile: np.ndarray,
    l_Welch: int | None = None,
    O_Welch: int | None = None,
    F_Welch: int | None = None,
    fs: float = 1.0,
    window: str = "hann",
    detrend: str = "constant",
    average: str = "median",
    scaling: str = "density",
    nu_min: float | None = None,
    nu_max: float = 0.05,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate the Hurst exponent using Welch's PSD estimator.

        l_Welch = floor(L/2)   (segment length)
        O_Welch = floor(L/4)   (overlap)
        F_Welch = L            (FFT length)

    Parameters
    ----------
    profile : np.ndarray
        1D profile (e.g., fractional Brownian motion).

    l_Welch : int, optional
        Segment length used by Welch (maps to scipy.signal.welch `nperseg`).
        Default: floor(L/2).

    O_Welch : int, optional
        Overlap between segments (maps to welch `noverlap`).
        Default: floor(L/4).

    F_Welch : int, optional
        FFT length (maps to welch `nfft`).
        Default: L.

    fs : float, optional
        Sampling frequency of the profile. Only affects frequency axis scaling.
        Default: 1.0.

    window : str, tuple or array-like, default="hann"
        Window passed to :func:`scipy.signal.welch`.

    detrend : str, callable or False, default="constant"
        Segment-wise detrending passed to SciPy.

    average : {"mean", "median"}, default="median"
        Aggregation of segment-wise periodograms.

    scaling : {"density", "spectrum"}, default="density"
        PSD scaling passed to SciPy.

    nu_min : float, optional
        Minimum spatial frequency used for fitting. Defaults to ``4/L``.
    nu_max : float, default=0.05
        Maximum spatial frequency used for fitting.

    Returns
    -------
    H_PSD : float
        PSD-based Hurst exponent estimate.
    frequencies : ndarray
        Positive frequencies used in the log--log regression.
    M_PSD : ndarray
        Welch spectral-density estimates used in the regression.

    Raises
    ------
    ValueError
        If the profile, Welch parameters, or frequency interval are invalid.
    RuntimeError
        If fewer than two positive spectral estimates lie in the fitting
        interval.

    Notes
    -----
    The fitted slope ``b_PSD`` is transformed according to
    ``H_PSD = -(b_PSD + 1)/2``. Zero-padding through ``F_Welch`` samples the
    spectrum on a finer grid but does not increase intrinsic resolution.
    """
    z = np.asarray(profile, dtype=float)
    if z.ndim != 1 or not np.all(np.isfinite(z)):
        raise ValueError("profile must be a one-dimensional array of finite values.")
    L = z.size

    # --- defaults exactly as stated in the text
    if l_Welch is None:
        l_Welch = L // 2
    if O_Welch is None:
        O_Welch = L // 4
    if F_Welch is None:
        F_Welch = L

    l_Welch = int(l_Welch)
    O_Welch = int(O_Welch)
    F_Welch = int(F_Welch)

    # --- sanity checks
    if not (2 <= l_Welch <= L):
        raise ValueError(f"l_Welch must be in [2, L]; got {l_Welch} with L={L}.")
    if not (0 <= O_Welch < l_Welch):
        raise ValueError(f"O_Welch must satisfy 0 <= O_Welch < l_Welch; got {O_Welch}, {l_Welch}.")
    if F_Welch < l_Welch:
        raise ValueError(f"F_Welch must be >= l_Welch; got {F_Welch} < {l_Welch}.")

    freqs, psd_values = welch(
        z,
        fs=fs,
        window=window,
        nperseg=l_Welch,
        noverlap=O_Welch,
        nfft=F_Welch,
        detrend=detrend,
        average=average,
        scaling=scaling,
    )

    if nu_min is None:
        nu_min = 4 / L
    if not 0 < nu_min < nu_max <= fs / 2:
        raise ValueError("Require 0 < nu_min < nu_max <= fs/2.")

    mask = (freqs >= nu_min) & (freqs <= nu_max)
    frequencies = freqs[mask]
    M_PSD = psd_values[mask]

    good = (frequencies > 0) & (M_PSD > 0)
    frequencies = frequencies[good]
    M_PSD = M_PSD[good]

    if frequencies.size < 2:
        raise RuntimeError("Not enough frequency points; adjust nu_min or nu_max.")

    b_PSD = np.polyfit(np.log(frequencies), np.log(M_PSD), 1)[0]

    # PSD slope relation for fBm
    H_PSD = -(b_PSD + 1.0) / 2.0
    return float(H_PSD), frequencies, M_PSD
