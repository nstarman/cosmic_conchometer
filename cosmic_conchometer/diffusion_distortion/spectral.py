# -*- coding: utf-8 -*-

"""Spectral Distortion."""

__all__ = ["SpectralDistortion"]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T
from functools import cached_property

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.cosmology.core import Cosmology
from classy import Class as CLASS

from scipy.special import binom, factorial
from scipy.interpolate import InterpolatedUnivariateSpline, CubicSpline

# PROJECT-SPECIFIC
from .core import DiffusionDistortionBase
from cosmic_conchometer.typing import ArrayLikeCallable
from cosmic_conchometer.utils import distances


##############################################################################
# PARAMETERS

I = int
rho0 = distances.rho0.value

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
    GgamBarCL : Callable
    PgamBarCL : Callable
    AkFunc: Callable or str or None (optional)

    """

    def __init__(
        self,
        cosmo: Cosmology,
        class_cosmo: CLASS,
        # TODO! temporary
        C1F2,
        C2F2,
        *,
        AkFunc: T.Union[str, ArrayLikeCallable, None] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            cosmo=cosmo, class_cosmo=class_cosmo, AkFunc=AkFunc, **kwargs
        )

        self.nmax = kwargs.get("nmax", 50)

        # ---------------------------------------
        # C-functions

        self.C1F2 = C1F2
        self.C2F2 = C2F2

        # ---------------------------------------
        # splines & related

        # get calculated quantities from CLASS
        thermo = class_cosmo.get_thermodynamics()
        z = thermo["z"][::-1]
        dkappadtau = thermo["kappa' [Mpc^-1]"] << 1 / u.Mpc  # g / P
        gbarCL = thermo["g [Mpc^-1]"][::-1] << 1 / u.Mpc

        # transform z to rho
        zeq = distances.z_of.matter_radiation_equality(cosmo=cosmo)
        rho = distances.rho_of.z(z, zeq=zeq)
        rho0x = rho - rho0

        # get the spline bounding box.
        # it can't be larger than the endpoints of the data
        bboxlimit = (min(rho0x).value, max(rho0x).value)
        bbox = kwargs.get("bbox", bboxlimit)
        if bbox[0] < bboxlimit[0] or bboxlimit[-1] < bbox[-1]:
            raise ValueError(f"bbox must be within {bboxlimit}")

        self._Nknots = Nknots = kwargs.get("Nknots", int(np.diff(bbox)))
        self._Delta = Delta = (bbox[-1] - bbox[0]) / (Nknots - 1)
        self._rho0x_knots = np.arange(bbox[0], bbox[-1] + Delta, Delta)

        # need to trim the data to fit in the bounding box
        inbbox = (bbox[0] <= rho0x) & (rho0x <= bbox[-1])
        rho = rho[inbbox]
        rho0x = rho0x[inbbox]
        gbarCL = gbarCL[inbbox]
        dkappadtau = dkappadtau[inbbox]

        # g-spline
        # done in 2 steps: 1) preparatory spline,
        #                  2) Cubic spline with predefined knot points
        prep_gspline = InterpolatedUnivariateSpline(
            rho0x.to_value(u.one),
            (self.lambda0 * gbarCL).to_value(u.one),
            ext=3,  # constant at the boundary, for evaluating at bbox[-1]
            bbox=bbox,
        )
        self._GgamBarCL_spl = CubicSpline(
            self.rho0x_knots, prep_gspline(self.rho0x_knots), extrapolate=False
        )

        # g/P-spline
        # done in 2 steps: 1) preparatory spline,
        #                  2) Cubic spline with predefined knot points
        prep_S_spl = InterpolatedUnivariateSpline(
            rho0x.to_value(u.one),
            (self.lambda0 * dkappadtau / rho).to_value(u.one),
            ext=3,  # constant at the boundary, for evaluating at bbox[-1]
            bbox=bbox,
        )
        self._S_spl = CubicSpline(
            self.rho0x_knots, prep_S_spl(self.rho0x_knots), extrapolate=False
        )

    # /def

    @property
    def Nknots(self):
        return self._Nknots

    @property
    def Delta(self):
        return self._Delta

    @property
    def rho0x_knots(self):
        return self._rho0x_knots

    @cached_property
    def rho0x_bbox(self):
        return (self._rho0x_knots[0], self._rho0x_knots[-1])

    # ===============================================================

    @u.quantity_input(freq=u.GHz, k=1 / u.Mpc)
    def prefactor(
        self,
        freq: u.Quantity,
        k: u.Quantity,
        real_AK: T.Optional[bool] = None,
    ) -> u.Quantity:
        r"""Combined Prefactor.

        .. math::

            \frac{\lambda_0^2 A(\vec{k})}{16\pi \bar{P}_{\gamma}^{CL}(\zeta_0)}
            \frac{h\nu/k_B T_0}{e^{-\frac{h\nu}{k_B T_0}}-1}

        Parameters
        ----------
        freq : `~astropy.units.Quantity`
            The frequency.
        k : `~astropy.units.Quantity`
            Units of inverse Mpc.

        Returns
        -------
        `~astropy.units.Quantity`
            units of [Mpc**2 * units[AkFunc]]

        Other Parameters
        ----------------
        real_AK : bool or None (optional)
            Whether to use the real (True) or imaginary (False) component of
            Ak or return the whole complex factor (None, default)

        """
        raise NotImplementedError("TODO!")

    #         reduced_energy = freq * const.h / (const.k_B * self.Tcmb0) << u.one
    #
    #         Ak = self.AkFunc(k)
    #         if real_AK is True:
    #             Ak = np.real(Ak)
    #         elif real_AK is False:
    #             Ak = np.imaginary(Ak)
    #
    #         prefactor = (
    #             (reduced_energy / np.expm1(-reduced_energy))
    #             * self.lambda0 ** 2
    #             * Ak
    #             / (16 * np.pi * self.PgamBarCL0)
    #         )
    #
    #         return prefactor

    # /def

    # ===============================================================

    def _l_sum(self) -> np.ndarray:  # TODO! descriptive name
        raise NotImplementedError("TODO!")

    def _i_sum(self) -> np.ndarray:  # TODO! descriptive name
        raise NotImplementedError("TODO!")

    # ----------------------------------------------------------

    def _C1F2_difference(
        self, bD: float, i_rhoES: I, *, n, p, q, v, m
    ) -> float:
        """Difference between C-1F2s.

        Parameters
        ----------
        bD : float
            :math:`\beta\Delta`
        i_rhoES : int
            :math:`\rho_{ES}` knot index.
        n, p, q, v, m : int
            Indices. # TODO! explanation

        """
        # t1 : coefficient for 1st C1F2
        t10 = (p + q + 1) * 1j * self.Delta if n != (p + q + v + 1) else 0
        t11 = rho0 * bD
        C1 = (t10 + t11) * self.C1F2(bD, i_rhoES, v, m)

        # t2 : coefficient for 2nd C1F2
        C2 = bD * (i_rhoES + 1) * self.C1F2(bD, i_rhoES, v + 1, m)

        return C1 - C2

    def _C1F2_nsummand(
        self, bD: float, i_rhoES: I, i_rho0S: I, *, n, p, q, v, m
    ) -> float:
        """C1F2-difference summand.

        Parameters
        ----------
        bD : float
            :math:`\beta\Delta`
        i_rhoES : int
            :math:`\rho_{ES}` knot index.
        n, p, q, v, m : int
            Indices. # TODO! explanation

        """
        t1 = factorial(m) * factorial(p + q) / factorial(n)
        t2 = -1j * np.power(-1j * bD, n - (p + q + v + 2))
        t3 = np.power(i_rho0S + 1, n - v) * np.power(i_rhoES + 1, v)
        t4 = t1 * t2 * t3

        return t4 * self._C1F2_difference(bD, i_rhoES, n=n, p=p, q=q, v=v, m=m)

    def _C1F2_nsum(self, bD: float, i_rhoES: I, i_rho0S: I, *, p, q, v, m):
        """C1F2-difference summand.

        Parameters
        ----------
        bD : float
            :math:`\beta\Delta`
        i_rhoES : int
            :math:`\rho_{ES}` knot index.
        p, q, v, m : int
            Indices. # TODO! explanation

        """
        return np.sum(
            self._C1F2_nsummand(bD, i_rhoES, i_rho0S, n=n, p=p, q=q, v=v, m=m)
            for n in range(p + q + v + 1, self.nmax + 1)
        )

    def _C1F2_vsummand(self, bD: float, i_rhoES: I, i_rho0S: I, *, p, q, v, m):
        """C1F2-difference summand.

        Parameters
        ----------
        bD : float
            :math:`\beta\Delta`
        i_rhoES : int
            :math:`\rho_{ES}` knot index.
        p, q, v, m : int
            Indices. # TODO! explanation

        """
        return (
            binom(v - m, v - m - q)
            * np.power(self.Delta, m + p + q + v + 2)
            * self._C1F2_nsum(bD, i_rhoES, i_rho0S, p=p, q=q, v=v, m=m)
        )

    def _C1F2_vsum(self, bD: float, i_rhoES: I, i_rho0S: I, *, p, q, m):
        """C1F2-difference summand.

        Parameters
        ----------
        bD : float
            :math:`\beta\Delta`
        i_rhoES : int
            :math:`\rho_{ES}` knot index.
        p, q, m : int
            Indices. # TODO! explanation

        """
        return np.sum(
            self._C1F2_vsummand(bD, i_rhoES, i_rho0S, p=p, q=q, m=m)
            for v in range(m + q, m + 3 + 1)
        )

    # TODO!! this is the wrong order for Eq. 127

    def _C1F2_pqsum(self, bD: float, i_rhoES: I, i_rho0S: I, *, m):
        """C1F2-difference summand.

        Parameters
        ----------
        bD : float
            :math:`\beta\Delta`
        i_rhoES : int
            :math:`\rho_{ES}` knot index.
        m : int
            Indices. # TODO! explanation

        """
        return np.sum(
            np.sum(
                self._C1F2_vsum(bD, i_rhoES, i_rho0S, p=p, q=q, m=m)
                for p in range(0, 3 + 1)
            )
            for q in range(0, 3 + 1)
        )

    def _C1F2_msummand(self, bD: float, i_rhoES: I, i_rho0S: I, *, m):
        """C1F2-difference summand.

        Parameters
        ----------
        bD : float
            :math:`\beta\Delta`
        i_rhoES : int
            :math:`\rho_{ES}` knot index.
        m : int
            Indices. # TODO! explanation

        """
        return (
            np.power(k * self.lambda0 * np.sin(thetakS), 2 * m)
            / np.power(2, m)
            / (2 * m + 3)
            / np.square(factorial(m))
        ) * _C1F2_pqsum(bD, i_rhoES, i_rho0S, m=m)

    def _C1F2_msum(
        self,
    ):
        """C1F2-difference summand.

        Parameters
        ----------
        bD : float
            :math:`\beta\Delta`
        i_rhoES : int
            :math:`\rho_{ES}` knot index.

        """
        return np.sum(
            self._C1F2_msummand(bD, i_rhoES, i_rho0S, m=m)
            for m in range(0, self.mmax + 1)
        )

    # def _C1F2_prefactor(self):
    #     self.Pspline(i) - self.Pspline(i + 1) * self.gspline(i - l)

    # ------------------------------------------------------------

    #     def _C2F2_difference(self, bD, l, n, p, q, v, i):
    #         """Difference between C-nogammas."""
    #         t0 = np.power(i-l, n-v) * np.power(l+1, v)
    #
    #         t10 = (p + q + 1) * 1j * self.Delta if n != (p + q + v + 1) else 0
    #         t11 = self.rho0 * bD
    #         t1 = t10 + t11
    #
    #         # is it upper-inclusive?
    #         t2Cs = (
    #             self._C2F2_difference_in_sum(bD, l=l, n=n, p=p, q=q, v=v, i=i)
    #             for r in range(v+1, n+1)
    #         )
    #
    #         t3 = bD * np.power(l + 1, n+1)
    #
    #         return t0 * t1 * self.C2F2(bD, l, v, m) - np.sum(t2Cs) * t3 * self.C2F2(bD, l, n+1, m)
    #
    #     def _C2F2_difference_in_sum(self, bD, l, n, p, q, v, i):
    #         t10 = (p + q + 1) * 1j * self.Delta if n != (p + q + v + 1) else 0
    #         t11 = self.rho0 * bD
    #         t1 = t10 + t11
    #
    #         t2 = binom(n-v, r-v) * np.power(i-l, n-r) * np.power(l+1, r) *
    #         (t1 - (r-v) * (i - l) / (n + 1 - r))
    #
    #         return t2 * self.C2F2(bD, l, r, m)

    # ===============================================================
    # Convenience Methods

    @u.quantity_input(freq=u.GHz, k=1 / u.Mpc, theta_kS=u.rad)
    def __call__(
        self,
        freq: u.Quantity,
        k: u.Quantity,
        theta_kS: u.Quantity,
        *,
        zeta_min: float = 0.0,
        zeta_max: float = np.inf,
        m_max: int = 100,
    ) -> u.Quantity:
        r"""Perform Calculation.

        The cross-terms in :math:`A(k)` with the integral cancel out.
        Only the real-with-real and imaginary-with-imaginary remain.

        Parameters
        ----------
        freq : `~astropy.units.Quantity`
            Frequency [GHz].
        k : `~astropy.units.Quantity`
            :math:`k` magnitude, [1/Mpc].
        theta_kS : `~astropy.units.Quantity`
            Angle between :math:`k` and scattering location, [radians].

        Returns
        -------
        spectral_distortion : `~astropy.units.Quantity`

        Other Parameters
        ----------------
        zeta_min, zeta_max : float (optional, keyword-only)
            Minimum / Maximum :math:`\zeta`
        m_max : int (optional, keyword-only)
            Maximum index in summation.

        """
        raise NotImplementedError("TODO!")

    compute = __call__
    # /def

    #######################################################


#     def plot_PgamBarCL(self, plot_times: bool = False) -> None:
#         # THIRD PARTY
#         import matplotlib.pyplot as plt
#
#         plt.scatter(
#             self._zeta_arr,
#             self.PgamBarCL(self._zeta_arr),
#             label=r"$\bar{P}_{\gamma}^{\mathrm{CL}}(\zeta)$",
#         )
#
#         # some interesting times
#         if plot_times:
#             plt.axvline(self.zeta0, label=r"$\zeta_0$", c="k", ls="-")
#             plt.axvline(1, label=r"$\zeta_{\mathrm{eq}}$", c="k", ls="--")
#             plt.axvline(self.zeta(1100), label=r"$\zeta_{R}$", c="k", ls=":")
#
#         plt.legend()
#
#         # return fig
#
#     # /def
#
#     def plot_GgamBarCL(self, plot_times: bool = False) -> None:
#         # THIRD PARTY
#         import matplotlib.pyplot as plt
#
#         plt.scatter(
#             self._zeta_arr,
#             self.GgamBarCL(self._zeta_arr),
#             label=r"$\bar{P}_{\gamma}^{\mathrm{CL}}(\zeta)$",
#         )
#
#         # some interesting times
#         if plot_times:
#             plt.axvline(self.zeta0, label=r"$\zeta_0$", c="k", ls="-")
#             plt.axvline(1, label=r"$\zeta_{\mathrm{eq}}$", c="k", ls="--")
#             plt.axvline(self.zeta(1100), label=r"$\zeta_{R}$", c="k", ls=":")
#
#         plt.legend()
#
#         # return fig
#
#     # /def


# /class

##############################################################################
# END
