"""Spectral Distortion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
from scipy.fft import fft, fftfreq, fftshift, fht, fhtoffset
from scipy.interpolate import (
    RectBivariateSpline,
)

__all__: list[str] = []

if TYPE_CHECKING:
    from cosmic_conchometer._typing import NDAf


@overload
def fft_P(
    spll: NDAf,
    sprp: NDAf,
    P: NDAf,
    *,
    sprp_lnpad: float = ...,
    full_output: Literal[False] = ...,  # https://github.com/python/mypy/issues/6580
    _dering: bool = ...,
) -> tuple[NDAf, NDAf, NDAf]:
    ...


@overload
def fft_P(
    spll: NDAf,
    sprp: NDAf,
    P: NDAf,
    *,
    sprp_lnpad: float = ...,
    full_output: Literal[True],
    _dering: bool = ...,
) -> tuple[NDAf, NDAf, NDAf, NDAf]:
    ...


def fft_P(
    spll: NDAf,
    sprp: NDAf,
    P: NDAf,
    *,
    sprp_lnpad: float = 8,
    full_output: bool = False,
    _dering: bool = True,
) -> tuple[NDAf, NDAf, NDAf] | tuple[NDAf, NDAf, NDAf, NDAf]:
    r"""FFT :math:`\mathcal{P}`.

    Parameters
    ----------
    spll : ndarray
        Ordered min to max.
    sprp : ndarray
        Ordered min to max.
    P : ndarray
        P.
    sprp_lnpad : float, optional keyword-only
        Log-paddding for sprp.
    full_output : bool, optional keyword-only
        Whether to return the full output.

    Returns
    -------
    qpll, qprp : NDArray
        Fourier versions of spll, sprp.
    Ptilde : NDArray
        Fourier-transformed P.
    Pqs : NDArray

    Other Parameters
    ----------------
    _dering : bool
        Whether to de-ring the fht offset.
    """
    # zero padding sprp ----
    # this is done to push the "ringing" of the FHT well below
    # min(qprp) = 2 pi / max(sprp)

    sprpmin, sprpmax = min(sprp), max(sprp)
    dln = np.diff(np.log(sprp))[0]  # TODO! check all close, not just 0th
    # TODO! cleaner make of sprp_large
    sprp_pad = np.exp(np.log(sprpmax) + np.arange(0, sprp_lnpad, dln)[1:])
    sprp_padded = np.concatenate((sprp, sprp_pad))

    P_padded = np.zeros((len(spll), len(sprp_padded)))
    P_padded[: len(spll), : len(sprp)] = P

    # ---- fft ----

    # Do the fft (auto-detects P is real)
    gtilde = fftshift(fft(P_padded[:, :], axis=0), axes=0)

    # `fft` assumed spll is in the range [0, N), not [smin, smax]
    minspll = min(spll)
    deltaspll = max(spll) - minspll
    N = P_padded.shape[0]
    freq = fftfreq(N, d=1 / N).astype(int)
    qpll = (2 * np.pi / deltaspll) * freq
    qpll = fftshift(qpll)

    Pqpllsprp = (deltaspll / N) * np.exp(-1j * minspll * qpll[:, None]) * gtilde[:, :]

    # ---- fht ----
    # Now do the FHT (real and imaginary)
    # The problem is that the FHT in scipy is of a real periodic input array,
    # which we don't have. We can make the array periodic by subtracting
    # off a function whose FHT is known. Note the function is complex.
    sprpmax = max(sprp_padded)  # (min is the same)
    pmin = Pqpllsprp[:, [0]]  # complex
    pmax = Pqpllsprp[:, [-1]]  # complex
    a_f = (pmax * sprpmax - pmin * sprpmin) / (sprpmax**2 - sprpmin**2)
    b_f = (
        sprpmax
        * sprpmin
        * (pmin * sprpmax - pmax * sprpmin)
        / (sprpmax**2 - sprpmin**2)
    )
    func_known_fht = a_f * sprp_padded + b_f / sprp_padded

    # The periodic version of P
    Phat = Pqpllsprp - func_known_fht  # \hat{P}(q_{||}, s_\perp)
    # The FHT inputs
    mu = bias = 0
    offset = np.log(2 * np.pi)
    if _dering:
        # optimal offset to prevent ringing
        offset = fhtoffset(dln, mu, initial=offset, bias=bias)

    # Calculating real and imaginary FHT on the periodic function
    real = fht(Phat.real, dln=dln, mu=mu, offset=offset, bias=bias)
    imag = fht(Phat.imag, dln=dln, mu=mu, offset=offset, bias=bias)

    # Getting q_\perp
    n = len(sprp_padded)
    jc = (n - 1) / 2
    rc = sprpmin * np.exp(jc * dln)  # sprp_j = rc exp[(j-jc) dln]  for any j
    kc = np.exp(offset) / rc
    js = np.arange(n)  # non-inclusive
    qprp = kc * np.exp((js - jc) * dln)

    # Need to add back the FHT of func_known_fht
    func_fht = -a_f / qprp[None, :] ** 3 + b_f / qprp[None, :]
    resfht_padded = (real + func_fht.real) + 1j * (imag + func_fht.imag)

    # Full Ptilde, undpadded!
    qprp = qprp[len(sprp_pad) :]
    resfht = resfht_padded[:, len(sprp_pad) :]
    Ptilde = resfht / qprp[None, :]

    if full_output:
        return qpll, qprp, Ptilde, Pqpllsprp[:, len(sprp_pad) :]
    return qpll, qprp, Ptilde


###############################################################################


def compute_fft(
    spll: NDAf,
    sprp: NDAf,
    Parr: NDAf | RectBivariateSpline,
    **kwargs: Any,
) -> tuple[NDAf, NDAf, NDAf, NDAf]:
    """Compute the FFT."""
    P = Parr(spll, sprp) if isinstance(Parr, RectBivariateSpline) else Parr

    qpll, qprp, Ptilde = fft_P(
        spll,
        sprp,
        P,
        full_output=False,
        sprp_lnpad=kwargs.pop("sprp_lnpad", 8),
        _dering=kwargs.pop("_dering", True),
    )

    spl = RectBivariateSpline(qpll, qprp, np.abs(Ptilde), kx=3, ky=3, s=0)
    correction = spl(0, 0, grid=False)  # should be 1.

    return qpll, qprp, Ptilde / correction, correction
