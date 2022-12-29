"""Spectral Distortion."""

from __future__ import annotations

# STDLIB
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast, overload

# THIRD-PARTY
import astropy.units as u
import numpy as np
import tqdm
from scipy.fft import fft, fftfreq, fftshift, fht, fhtoffset
from scipy.interpolate import (
    CubicSpline,
    InterpolatedUnivariateSpline,
    RectBivariateSpline,
)

# LOCAL
from .utils import rho2_of_rho1

if TYPE_CHECKING:
    # THIRD-PARTY

    # LOCAL
    from cosmic_conchometer._typing import NDAf, scalarT

__all__ = [
    "ComputePspllSprp",
    "fft_P",
]

##############################################################################
# TYPING

ArgsT: TypeAlias = tuple[
    "NDAf", "NDAf", "NDAf", float, float, "NDAf", "NDAf", "NDAf", "NDAf", "NDAf", "NDAf"
]

##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class ComputePspllSprp:
    """Compute P(spll, sprp) for a given spectral distortion."""

    lambda0: float  # [Mpc]
    rho_domain: tuple[float, float]  # (min, max)

    spl_ln_gbarCL: InterpolatedUnivariateSpline  # ([dimensionless]) -> [Mpc]
    spl_ln_PbarCL: InterpolatedUnivariateSpline  # ([dimensionless]) -> [Mpc]

    @property
    def prefactor(self) -> float | scalarT:
        """The prefactor for the integral."""
        return cast(
            "float | scalarT",
            3
            * self.lambda0**2
            / (16 * np.exp(self.spl_ln_PbarCL(self.rho_domain[-1]))),
        )

    # ------------------------------------
    # Integrals between knots (vectorized)

    def _integral0(self, args: ArgsT, /) -> NDAf:
        xi, xj, _, pllrO, sprp, _, _, xjs2, xis2, diffatan, _ = args
        t0 = (
            -sprp * ((xj + pllrO) / xjs2 - (xi + pllrO) / xis2)  # delta 1st term
            + 3 * diffatan  # delta 2nd term
        )
        return t0

    def _integral1(self, args: ArgsT, /) -> NDAf:
        pllrO, sprp, xjpllrO, xipllrO, xjs2, xis2, diffatan, difflog = args[3:]
        t1: NDAf = (
            +(sprp * pllrO * xjpllrO + sprp**3) / xjs2
            - (sprp * pllrO * xipllrO + sprp**3) / xis2
            - 3 * pllrO * diffatan
            + 2 * sprp * difflog
        )
        return t1

    def _integral2(self, args: ArgsT, /) -> NDAf:
        xi, xj, dx, pllrO, sprp, xjpllrO, xipllrO, xjs2, xis2, diffatan, difflog = args
        t2: NDAf = (
            +5 * sprp * dx
            - sprp * xjpllrO * xj**2 / xjs2
            + sprp * xipllrO * xi**2 / xis2
            + (3 * pllrO**2 - 5 * sprp**2) * diffatan
            - 4 * pllrO * sprp * difflog
        )
        return t2

    def _integral3(self, args: ArgsT, /) -> NDAf:
        xi, xj, dx, pllrO, sprp, xjpllrO, xipllrO, xjs2, xis2, diffatan, difflog = args
        t3: NDAf = (
            +3 * sprp * ((xj**2 - xi**2) - 3 * pllrO * dx)
            - sprp * xjpllrO * xj**3 / xjs2
            + sprp * xipllrO * xi**3 / xis2
            - 3 * pllrO * (pllrO**2 - 5 * sprp**2) * diffatan
            + 6 * sprp * (pllrO**2 - 0.5 * sprp**2) * difflog
        )
        return t3

    # ------------------------------------

    def _visibilities_factor(self, rho: NDAf, spll: scalarT, sprp: scalarT) -> NDAf:
        """Returns log [g/P](rho_1)*g(rho2)."""
        rho2 = rho2_of_rho1(rho, spll, sprp, maxrho_domain=self.rho_domain[-1])
        vf: NDAf = np.exp(
            self.spl_ln_gbarCL(rho) - self.spl_ln_PbarCL(rho) + self.spl_ln_gbarCL(rho2)
        )
        return vf

    def _make_visibilities_spline(
        self, rho: NDAf, spll: scalarT, sprp: scalarT
    ) -> CubicSpline:
        gs = self._visibilities_factor(rho, spll, sprp)
        gs[gs < 0] = 0
        gs[np.isnan(gs)] = 0  # shouldn't happen
        return CubicSpline(rho, gs)

    def _global_poly_coeffs_from_ppoly_coeffs(
        self, spl: CubicSpline
    ) -> tuple[float, float, float, float]:
        """Convert PPoly coefficients to global coefficients.

        ::
            c3(x-xi)^3 + c2(x-xi)^2 + c1(x-xi) + c0
            = p3 x^2 + p2 x^2 + p1 x + p0.

            p3 = c3
            p2 = -3 c3 xi + c2
            p1 = 3 c3 xi^2 - 2 c2 xi + c1
            p0 = -c3 xi^3 + c2 xi^2 - c1 xi + c0
        """
        xi = spl.x[:-1]
        c3, c2, c1, c0 = spl.c

        p3 = c3
        p2 = -3 * c3 * xi + c2
        p1 = 3 * c3 * xi**2 - 2 * c2 * xi + c1
        p0 = -c3 * xi**3 + c2 * xi**2 - c1 * xi + c0
        return p3, p2, p1, p0

    def _setup(self, rho: NDAf, spll: float, sprp: float) -> ArgsT:
        xi = rho[:-1]  # (N-1,)
        xj = rho[1:]  # (N-1,)
        dx = xj - xi  # (N-1,)
        pllrO = spll - self.rho_domain[-1]
        xipllrO = xi + pllrO
        xjpllrO = xj + pllrO
        xis2 = xipllrO**2 + sprp**2
        xjs2 = xjpllrO**2 + sprp**2

        # Trick for difference of arctans with same denominator
        # beta - alpha = arctan( (tan(beta)-tan(alpha)) / (1 + tan(alpha)tan(beta)) )
        #              = arctan( c(b - a) / (c^2 + a*b ) )
        diffatan = np.arctan2(sprp * dx, sprp**2 + xipllrO * xjpllrO)

        # Log term
        difflog = np.log(xjs2 / xis2)

        args = xi, xj, dx, pllrO, sprp, xjpllrO, xipllrO, xjs2, xis2, diffatan, difflog
        return args

    def __call__(self, rho: NDAf, spll: float, sprp: float) -> NDAf:
        r"""Evaluate :math:`\mathcal{P}(s_{||}, s_\perp)`.

        Parameters
        ----------
        rho: NDArray
            Rho.
        spll, sprp: float
            spll, sprp.

        Returns
        -------
        NDArray
        """
        _rho = rho.to_value(u.one) if isinstance(rho, u.Quantity) else rho
        spl = self._make_visibilities_spline(_rho, spll, sprp)  # units of inverse Mpc
        p3, p2, p1, p0 = self._global_poly_coeffs_from_ppoly_coeffs(spl)

        args = self._setup(_rho, spll, sprp)

        t0 = self._integral0(args)
        t1 = self._integral1(args)
        t2 = self._integral2(args)
        t3 = self._integral3(args)

        P: NDAf = self.prefactor * np.sum(p0 * t0 + p1 * t1 + p2 * t2 + p3 * t3)
        return P

    # ==========================================================================
    # On a grid

    def _make_rho(
        self,
        spll: scalarT,
        sprp: scalarT,
        *,
        n_sprp: int,
        n_rho_center: int,
        n_rho_lr: int,
    ) -> NDAf:
        # Center region
        center = self.rho_domain[-1] - spll
        center_lower: scalarT = max(center - n_sprp * np.abs(sprp), self.rho_domain[0])
        center_upper: scalarT = min(center + n_sprp * np.abs(sprp), self.rho_domain[1])

        # Add lower rho, if center doesn't hit lower bound
        rho_lower: NDAf
        if center_lower == self.rho_domain[0]:
            rho_lower = np.array([])
        else:
            rho_lower = np.linspace(
                self.rho_domain[0] + 1e-5, center_lower, num=n_rho_lr
            )

        rho_center = np.linspace(center_lower, center_upper, num=n_rho_center)

        # Add upper rho, if center doesn't hit upper bound
        rho_upper: NDAf
        if center_upper == self.rho_domain[1]:
            rho_upper = np.array([])
        else:
            rho_upper = np.linspace(center_upper, self.rho_domain[1], num=n_rho_lr)

        # TODO: not need the `unique` check
        rho: NDAf = np.unique(
            np.concatenate((rho_lower[:-1], rho_center, rho_upper[1:]))
        )
        return rho

    def compute_on_grid(
        self,
        Spll: NDAf,
        Sprp: NDAf,
        *,
        n_sprp: int = 15,
        n_rho_center: int = 1_000,
        n_rho_lr: int = 1_000,
    ) -> tuple[NDAf, float]:
        """Compute on a grid.

        Parameters
        ----------
        Spll, Sprp : NDArray
            Spll is columnar, Sprp is row-wise.

        n_sprp : int, optional keyword-only
            Number of sprp
        n_rho_center : int, optional keyword-only
            Number of rho in the center array.
        n_rho_lr : int, optional keyword-only
            Number of rho lef and right arrays.

        Returns
        -------
        Parr : NDarray or RectBivariateSpline
        tot_prob_correction : float
        """
        Parr = np.zeros_like(Spll)
        shape = Parr.shape

        for i, j in tqdm.tqdm(
            itertools.product(range(shape[0]), range(shape[1])),
            total=shape[0] * shape[1],
        ):
            rho = self._make_rho(
                Spll[i, j],
                Sprp[i, j],
                n_sprp=n_sprp,
                n_rho_center=n_rho_center,
                n_rho_lr=n_rho_lr,
            )
            Parr[i, j] = self(rho, Spll[i, j], Sprp[i, j])

        _spl = RectBivariateSpline(Spll[:, 0], Sprp[0, :], Parr, kx=3, ky=3, s=0)
        correction = _spl.integral(
            Spll.min(), Spll.max(), Sprp.min(), Sprp.max()
        )  # should be 1.

        return Parr / correction, correction

    def compute_fft(
        self,
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


########################################################################


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
) -> (tuple[NDAf, NDAf, NDAf] | tuple[NDAf, NDAf, NDAf, NDAf]):
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
    # ---- zero padding sprp ----
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
