# -*- coding: utf-8 -*-

"""Utilities for :mod:`cosmic_conchometer.diffusion_distortion`"""

# __all__ = [
#     # functions
#     "",
#     # other
#     "",
# ]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T

# THIRD PARTY
import numpy as np
from scipy.fft import fft, fftfreq, fht, ifftshift, fhtoffset


##############################################################################
# CODE
##############################################################################


def fft_sP(spll: np.ndarray, sprp: np.ndarray, sP: np.ndarray, *, offset: float=0.0,
           full_output=False
) -> T.Union[T.Tuple[np.ndarray, np.ndarray, np.ndarray], T.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Fourier Transform :math:`P(s_{||}, s_{\perp})`.

    This is done by Fourier transforming in :math:`s_{||}` and Hankel
    transforming in :math:`s_{\perp}`.

    Parameters
    ----------
    spll : (N, ) ndarray
    sprp : (M, ) ndarray
    sP : (N, M) ndarray
        :math:`s_{\perp} P(s_{||}, s_{\perp})`
    offset : float, optional keyword-only
        Offset of the uniform logarithmic spacing of the output array.
        The choice is modified by `scipy.fft.fhtoffset` to be optimal for 
        ``dln`` (from ``sprp``) and mu and bias of 0.
    full_output : bool, optional keyword-only
        Whether to also return the intermediate FFT (before the Hankel transform)

    Returns
    -------
    qpll : (N, ) ndarray
        The Fourier conjugate of ``spll``.
    qprp : (M, ) ndarray
        The Fourier conjugate of ``sprp``.
    Ptilde : (N, M) ndarray
        The Fourier transform of :math:`P(s_{||}, s_{\perp})` in both
        :math:`s_{||}` and :math:`s_{\perp}`.
    fft : (N, M) ndarray
        The Fourier transform of :math:`P(s_{||}, s_{\perp})` in :math:`s_{||}`.
        Only returned if ``full_output`` is `True`.
    """
    # Do the fft (auto-detects sP is real)
    gtilde = fft(sP[:, :], axis=0)

    # `fft` assumes spll is in the range [0, N), not [smin, smax]
    minspll = min(spll)
    deltaspll = max(spll) - minspll
    N = sP.shape[0]
    freq = fftfreq(N, d=1/N).astype(int)
    qpll = (2 * np.pi / deltaspll) * freq
    resfft = (deltaspll / N) * np.exp(-1j * minspll * qpll[:, None]) * gtilde[:, :]

    # Now do FHT (real and complex)
    dln = np.diff(np.log(sprp))[0]  # TODO! check all close, not just 0th
    offset = fhtoffset(dln, 0, initial=offset, bias=0)  # optimal offset

    jc = (len(sprp) + 1) / 2
    rc = sprp[0] * np.exp(jc * dln)  # sprp_j = rc exp[(j-jc) dln]  for any j
    kc = np.exp(offset) / rc
    js = np.arange(len(sprp))
    qprp = kc * np.exp((js - jc) * dln)

    real = fht(resfft.real, dln=dln, mu=0, offset=offset, bias=0)
    imag = fht(resfft.imag, dln=dln, mu=0, offset=offset, bias=0)
    resfht = real + 1j * imag
    Ptilde = 2 * np.pi * resfht / qprp[None, :]

    if full_output:
        return qpll, qprp, Ptilde, 2 * np.pi * resfft
    return qpll, qprp, Ptilde


##############################################################################
# END
