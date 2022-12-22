"""Spectral Distortion."""

##############################################################################
# IMPORTS

from __future__ import annotations
from dataclasses import dataclass
import itertools

# STDLIB
import typing as T
import warnings


import astropy.units as u
import numpy as np
from astropy.cosmology.core import Cosmology
from classy import Class as CLASS
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import CubicSpline
import tqdm
from scipy.fft import fft, fftfreq, fftshift, fht, fhtoffset

# PROJECT-SPECIFIC
from .core import DiffusionDistortionBase


__all__ = [
    "SpectralDistortion", "ComputePspllSprp", "fft_P",
]

##############################################################################
# CODE
##############################################################################


class SpectralDistortion(DiffusionDistortionBase):
    """Spectral Distortion.

    Parameters
    ----------
    cosmo : :class:`~astropy.cosmology.core.Cosmology`
        The cosmology
    class_cosmo : :class:`~classy.Class`
    AkFunc: Callable or str or None (optional, keyword-only)
    **kwargs
        Ignored.
    """

    def __init__(
        self,
        cosmo: Cosmology,
        class_cosmo: CLASS,
        # *,
        # AkFunc: T.Union[str, ArrayLikeCallable, None] = None,
        meta: T.Mapping | None = None,
        **kwargs: T.Any,
    ) -> None:
        super().__init__(
            cosmo=cosmo,
            class_cosmo=class_cosmo,
            # AkFunc=AkFunc,
            meta=meta,
            **kwargs,
        )

        # ---------------------------------------
        # splines & related

        rho = self.class_rho

        self._spl_gbarCL = IUS(rho, self.class_g, ext=1)
        # ext 1 makes stuff 0  # TODO! does this introduce a discontinuity?

        # self._spl_PbarCL = IUS(rho, self.class_P, ext=2)
        # instead, to get the right normalization we will define PbarCL from
        # an integral over gbarCL
        integral = [
            self._spl_gbarCL.integral(a, b) for a, b in zip(rho[:-1], rho[1:])
        ]
        class_P = self.lambda0.to_value(u.Mpc) * np.concatenate(
            ([0], np.cumsum(integral))
        )
        _spl_PbarCL = IUS(rho, class_P, ext=2)

        # normalize the spline
        a, b = self.rhovalid
        prob = (
            self.lambda0.to_value(u.Mpc)
            * self._spl_gbarCL.integral(a, b)
            / _spl_PbarCL(b)
        )
        self._spl_integral_norm = 1 / prob

        self._spl_PbarCL = IUS(rho, class_P / prob, ext=2)


@dataclass
class ComputePspllSprp:
    """Compute P(spll, sprp) for a given spectral distortion."""

    sd: SpectralDistortion

    @property
    def prefactor(self):
        return 3 * self.sd.lambda0**2 / (8 * self.sd._spl_PbarCL(self.sd.maxrhovalid))
    
    # ------------------------------------
    # Integrals between knots (vectorized)
    
    def _integral0(self, rhos, spll, sprp, args):
        xi, xj, dx, spllrO, xjspllrO, xispllrO, xjs2, xis2, deltaarctan, deltalog = args
        t0 = (
            - sprp * ((xj + spllrO) / xjs2 - (xi + spllrO) / xis2)  # delta 1st term
            + 3 * deltaarctan  # delta 2nd term
        )
        return 0.5 * t0

    def _integral1(self, rhos, spll, sprp, args):
        xi, xj, dx, spllrO, xjspllrO, xispllrO, xjs2, xis2, deltaarctan, deltalog = args
        t1 = (
            + (sprp * spllrO * xjspllrO + sprp**3) / xjs2
            - (sprp * spllrO * xispllrO + sprp**3) / xis2
            - 3 * spllrO * deltaarctan
            + 2 * sprp * deltalog
        )
        return 0.5 * t1

    def _integral2(self, rhos, spll, sprp, args):
        xi, xj, dx, spllrO, xjspllrO, xispllrO, xjs2, xis2, deltaarctan, deltalog = args
        t2 = (
            + 5 * sprp * dx
            - sprp * xjspllrO * xj**2 / xjs2
            + sprp * xispllrO * xi**2 / xis2
            + (3 * spllrO**2 - 5 * sprp**2) * deltaarctan
            - 4 * spllrO * sprp * deltalog
        )
        return 0.5 * t2


    def _integral3(self, rhoss, spll, sprp, args):
        xi, xj, dx, spllrO, xjspllrO, xispllrO, xjs2, xis2, deltaarctan, deltalog = args
        t3 = (
            + 3 * sprp * ((xj**2 - xi**2) - 3 * spllrO * dx)
            - sprp * xjspllrO * xj**3 / xjs2
            + sprp * xispllrO * xi**3 / xis2
            - 3 * spllrO * (spllrO**2 - 5 * sprp**2) * deltaarctan
            + 6 * sprp * (spllrO**2 - 0.5*sprp**2) * deltalog
        )
        return 0.5 * t3
    
    # ------------------------------------

    def _visibilities_factor(self, rho: np.floating | np.ndarray, spll: np.floating, sprp: np.floating) -> np.floating | np.ndarray:
        """Returns [g/P](rho_1)*g(rho2)"""
        rho2 = rho - np.sqrt((spll + rho - self.sd.maxrhovalid)**2 + sprp**2)
        return self.sd._spl_gbarCL(rho) / self.sd._spl_PbarCL(rho) * self.sd._spl_gbarCL(rho2)

    def _make_visibilities_spline(self, rho: np.floating, spll: np.floating, sprp: np.floating):
        gs = self._visibilities_factor(rho, spll, sprp)
        gs[gs < 0] = 0
        gs[np.isnan(gs)] = 0  # when denominator is 0, but numerators are 0
        return CubicSpline(rho, gs)

    def _global_poly_coeffs_from_ppoly_coeffs(self, ceoffs: ndarray, xi: ndarray) -> tuple:
        """
        ::
            c3(x-xi)^3 + c2(x-xi)^2 + c1(x-xi) + c0
            = p3 x^2 + p2 x^2 + p1 x + p0
            
            p3 = c3
            p2 = -3 c3 xi + c2
            p1 = 3 c3 xi^2 - 2 c2 xi + c1
            p0 = -c3 xi^3 + c2 xi^2 - c1 xi + c0
        """
        c3, c2, c1, c0 = ceoffs
        p3 = c3
        p2 = -3*c3*xi + c2
        p1 = 3 * c3 * xi**2 - 2 * c2 * xi + c1
        p0 = -c3 * xi ** 3 + c2 * xi ** 2 - c1 * xi + c0
        return p3, p2, p1, p0

    def _setup(self, rho: np.ndarray, spll: float, sprp: float) -> tuple:
        xi = rho[:-1]  # (N-1,)
        xj = rho[1:]  # (N-1,)
        dx = xj - xi  # (N-1,)
        spllrO = spll - self.sd.maxrhovalid
        xispllrO = xi + spllrO
        xjspllrO = xj + spllrO
        xis2 = xispllrO ** 2 + sprp**2
        xjs2 = xjspllrO ** 2 + sprp**2

        # Trick for difference of arctans with same denominator
        # alpha = arctan(a/c)
        # beta = arctan(b/c)
        # beta - alpha = arctan( (tan(beta)-tan(alpha)) / (1 + tan(alpha)tan(beta)) )
        #              = arctan( c(b - a) / (c^2 + a*b ) )
        deltaarctan = np.arctan2(sprp * dx, sprp**2 + xispllrO * xjspllrO)

        # Log term
        deltalog = np.log(xjs2 / xis2)

        return xi, xj, dx, spllrO, xjspllrO, xispllrO, xjs2, xis2, deltaarctan, deltalog

    def __call__(self, rho, spll, sprp):
        spl = self._make_visibilities_spline(rho, spll, sprp)  # units of inverse Mpc
        p3, p2, p1, p0 = self._global_poly_coeffs_from_ppoly_coeffs(spl.c, spl.x[:-1])
        
        args = self._setup(rho, spll, sprp)

        t0 = self._integral0(rho, spll, sprp, args)
        t1 = self._integral1(rho, spll, sprp, args)
        t2 = self._integral2(rho, spll, sprp, args)
        t3 = self._integral3(rho, spll, sprp, args)

        return self.prefactor.to_value(u.Mpc**2) * np.sum(p0 * t0 + p1 * t1 + p2 * t2 + p3 * t3)

    # ==========================================================================
    # On a grid

    def _make_rho(self, spll, sprp, n_sprp=10, n_rho_center=int(1e3), n_rho_lr=int(5e2)):
        # Center region
        center = self.sd.maxrhovalid - spll
        center_lower = max(center - n_sprp * np.abs(sprp), self.sd.rhovalid[0])
        center_upper = min(center + n_sprp * np.abs(sprp), self.sd.rhovalid[1])

        # Add lower rho, if center doesn't hit lower bound
        if center_lower == self.sd.rhovalid[0]:
            rho_lower = []
        else:
            rho_lower = np.linspace(self.sd.rhovalid[0] + 1e-5, center_lower, num=n_rho_lr)

        rho_center = np.linspace(center_lower, center_upper, num=n_rho_center)

        # Add upper rho, if center doesn't hit upper bound
        if center_upper == self.sd.rhovalid[1]:
            rho_upper = []
        else:
            rho_upper = np.linspace(center_upper, self.sd.rhovalid[1], num=n_rho_lr)

        # return np.concatenate((rho_lower[:-1], rho_center, rho_upper[1:]))
        rho = np.concatenate((rho_lower[:-1], rho_center, rho_upper[1:]))
        return np.unique(rho)  # TODO! not need this step
        
        

    def compute_on_grid(self, Spll, Sprp, n_sprp: int = 15, n_rho_center: int = 1_000, n_rho_lr: int = 1_000):
        result = np.zeros_like(Spll)
        shape = result.shape

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in multiply")

            for i, j in tqdm.tqdm(itertools.product(range(shape[0]), range(shape[1])), total=shape[0]*shape[1]):
                rho = self._make_rho(Spll[i,j], Sprp[i,j],
                            n_sprp=n_sprp, n_rho_center=n_rho_center, n_rho_lr=n_rho_lr)

                result[i, j] = self(rho, Spll[i, j], Sprp[i, j])



def fft_P(spll: np.ndarray, sprp: np.ndarray, P: np.ndarray,
          *, sprp_lnpad: float = 8, _dering: bool=True
) -> T.Union[T.Tuple[np.ndarray, np.ndarray, np.ndarray], T.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Parameters
    ----------
    spll : ndarray
        Ordered min to max.
    sprp : ndarray
        Ordered min to max.
    P : ndarray
        P.
    sprp_lnpad : float, optional
    """
    # ---- zero padding sprp ----
    # this is done to push the "ringing" of the FHT well below
    # min(qprp) = 2 pi / max(sprp)

    sprpmin, sprpmax = min(sprp), max(sprp)
    dln = np.diff(np.log(sprp))[0]  # TODO! check all close, not just 0th
    # TODO! cleaner make of sprp_large
    sprp_pad = np.exp(np.log(sprpmax) + np.arange(0, sprp_lnpad, dln)[1:])
    sprp_padded = np.concatenate((sprp, sprp_pad))

    P_padded = np.zeros((len(spll), len(sprp_padded)))
    P_padded[:len(spll), :len(sprp)] = P

    # ---- fft ----

    # Do the fft (auto-detects P is real)
    gtilde = fftshift(fft(P_padded[:, :], axis=0), axes=0)

    # `fft` assumed spll is in the range [0, N), not [smin, smax]
    minspll = min(spll)
    deltaspll = max(spll) - minspll
    N = P_padded.shape[0]
    freq = fftfreq(N, d=1/N).astype(int)
    qpll = (2 * np.pi / deltaspll) * freq
    qpll = fftshift(qpll)

    Pqpllsprp = (deltaspll / N) * np.exp(-1j * minspll * qpll[:, None]) * gtilde[:, :]

    # Now do the FHT (real and imaginary)
    # The problem is that the FHT in scipy is of a real periodic input array,
    # which we don't have. We can make the array periodic by subtracting
    # off a function whose FHT is known. Note the function is complex.
    sprpmax = max(sprp_padded)  # (min is the same)
    pmin = Pqpllsprp[:, [0]]  # complex
    pmax = Pqpllsprp[:, [-1]]  # complex
    a_f = (pmax * sprpmax - pmin * sprpmin) / (sprpmax**2 - sprpmin**2)
    b_f = sprpmax * sprpmin * (pmin * sprpmax - pmax * sprpmin) / (sprpmax**2 - sprpmin**2)
    func_known_fht = a_f * sprp_padded + b_f / sprp_padded

    # The periodic version of P
    Phat = Pqpllsprp - func_known_fht # \hat{P}(q_{||}, s_\perp)
    # The FHT inputs
    mu = bias = 0
    offset = np.log(2 * np.pi)
    if _dering:
        print(f"initial: {offset}", end=" | ")
        offset = fhtoffset(dln, mu, initial=offset, bias=bias)  # optimal offset to prevent ringing
        print(f"final: {offset}")

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
    func_fht = -a_f / qprp[None, :]**3 + b_f / qprp[None, :]
    resfht_padded = (real + func_fht.real) + 1j * (imag + func_fht.imag)

    # Full Ptilde, undpadded!
    qprp = qprp[len(sprp_pad):]
    resfht = resfht_padded[:, len(sprp_pad):]
    Ptilde = resfht / qprp[None, :]

    return qpll, qprp, Ptilde, Pqpllsprp[:, len(sprp_pad):]
