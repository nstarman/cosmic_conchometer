"""Spectral Distortion."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, cast

import astropy.units as u
import numpy as np
import tqdm
from scipy.interpolate import (
    CubicSpline,
    InterpolatedUnivariateSpline,
    RectBivariateSpline,
)

from cosmic_conchometer.temperature_diffusion_spectra_distortion.prob_2ls.utils import (
    cubic_global_coeffs_from_ppoly_coeffs,
)
from cosmic_conchometer.temperature_diffusion_spectra_distortion.utils import (
    rho2_of_rho1,
)

if TYPE_CHECKING:
    from cosmic_conchometer._typing import NDAf, scalarT

__all__ = ["ComputePspllSprp"]

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
        return (
            -sprp * ((xj + pllrO) / xjs2 - (xi + pllrO) / xis2)  # delta 1st term
            + 3 * diffatan  # delta 2nd term
        )

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
        """Return log [g/P](rho_1)*g(rho2)."""
        rho2 = rho2_of_rho1(rho, spll, sprp, maxrho=self.rho_domain[-1])
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

        return xi, xj, dx, pllrO, sprp, xjpllrO, xipllrO, xjs2, xis2, diffatan, difflog

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
        p3, p2, p1, p0 = cubic_global_coeffs_from_ppoly_coeffs(spl)

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
        # the correction should be 1.
        correction = _spl.integral(Spll.min(), Spll.max(), Sprp.min(), Sprp.max())

        return Parr / correction, correction
