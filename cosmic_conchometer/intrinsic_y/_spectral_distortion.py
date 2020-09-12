# -*- coding: utf-8 -*-

"""Spectral Distortion."""

__all__ = [
    "SpectralDistortion",
]


##############################################################################
# IMPORTS

import typing as T

import astropy.constants as const
import astropy.units as u
import numpy as np
import scipy.integrate as integ
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.special import jv as besselJ

from .core import IntrinsicDistortionBase, ArrayLike_Callable

##############################################################################
# PARAMETERS

IUSType = T.Callable[[T.Union[float, np.ndarray]], np.ndarray]

QuantityType = T.TypeVar("Quantity", u.Quantity, u.SpecificTypeQuantity)


##############################################################################
# CODE
##############################################################################


class SpectralDistortion(IntrinsicDistortionBase):
    r"""Spectral Distortion.

    .. math::

        \frac{I^{(sd)}(\nu,\hat{n})}{B_\nu(\nu, T_0)} =
        \left(
        \frac{\lambda_0^2 A(\vec{k})}{16\pi \bar{P}_{\gamma}^{CL}(\zeta_0)}
        \frac{h\nu/k_B T_0}{e^{-\frac{h\nu}{k_B T_0}}-1}
        \!\!\int_{\zeta_O}^{\infty} \!\!\!\! \rm{d}{\zeta_S}
            \frac{\bar{g}_\gamma^{CL}(\zeta_S) e^{ikr_{S}\hat{k}\cdot\hat{z}}}
                 {\bar{P}_{\gamma}^{CL}(\zeta_S)\sqrt{(1+\zeta_S)\zeta_S^3}}
        \!\!\left.\int_{\zeta_S}^{\infty}\right\vert_{\Omega_{SE}}\!\!\!\!\!\!\!
            \rm{d}{\zeta_E} \frac{\bar{g}_\gamma^{CL}(\zeta_E)}
                                 {\sqrt{(1+\zeta_E)\zeta_E^3}}
        2\left(\frac{2\pi}{k r_{SE}\cos\theta_{kS}}\right)^{3/2}
        \right.

        \left.
        \sum_{m=0}^\infty \frac{-1^m}{2^{m}m!} \left(k r_{SE} \sin\theta_{kS}\tan\theta_{kS}\right)^m
        \left[(2+m) J_{\frac{3}{2}+m}(k r_{SE} \cos\theta_{kS}) - k r_{SE} \cos\theta_{kS} J_{\frac{5}{2}+m}(k r_{SE} \cos\theta_{kS}) \right]
        \right)

    Parameters
    ----------
    cosmo : :class:`~astropy.cosmology.core.Cosmology`
    class_cosmo : :class:`~classy.Class`
    GgamBarCL : Callable
    PgamBarCL : Callable
    AkFunc: Callable or str or None, optional
    integration_method : Callable

    """

    def __init__(
        self,
        cosmo,
        class_cosmo,
        *,
        AkFunc: T.Union[str, ArrayLike_Callable, None] = None,
        integration_method=integ.quad,
    ):
        """Spectral Distortion."""
        super().__init__(
            cosmo,
            class_cosmo,
            AkFunc=AkFunc,
            integration_method=integration_method,
        )

        # calculated quantities
        thermo = self.class_cosmo.get_thermodynamics()

        # TODO units
        self.PgamBarCL: IUSType = IUS(self._zeta_arr, thermo["exp(-kappa)"])
        self.GgamBarCL: IUSType = IUS(self._zeta_arr, thermo["g [Mpc^-1]"])

        self.PgamBarCL0: float = self.PgamBarCL(self.zeta0)

        # vectorized angular_summand
        self.angular_summand = np.vectorize(self._angular_summand)

    # /def

    # ------------------------------

    @u.quantity_input(freq=u.GHz, k=1 / u.Mpc)
    def prefactor(
        self, freq: QuantityType, k: QuantityType,
    ):
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

        """
        reduced_energy = freq * const.h / (const.k_B * self.Tcmb0) << u.one
        prefactor = (
            (reduced_energy / np.expm1(-reduced_energy))
            * self.lambda0 ** 2
            * self.AkFunc(k)
            / (16 * np.pi * self.PgamBarCL0)
        )

        return prefactor

    # /def

    # ------------------------------

    @staticmethod
    def _angular_summand(
        i: int, rES: float, k: float, theta_kS: float,
    ):
        r"""Angular Summation.

        .. math::

            2\left(\frac{2\pi}{k r_{SE}\cos\theta_{kS}}\right)^{3/2}
            \sum_{m=0}^\infty \frac{-1^m}{2^{m}m!} \left(k r_{SE} \sin\theta_{kS}\tan\theta_{kS}\right)^m
            \left[(2+m) J_{\frac{3}{2}+m}(k r_{SE} \cos\theta_{kS}) - k r_{SE} \cos\theta_{kS} J_{\frac{5}{2}+m}(k r_{SE} \cos\theta_{kS}) \right]

        Parameters
        ----------
        i: int
            Summation index.
        rES : float
            Distance from :math:`\zeta_E` to :math:`\zeta_S`, [Mpc].

        k : float
            :math:`k` magnitude, [1/Mpc].
        theta_kS : float
            Angle between :math:`k` and scattering location, [radians].

        Returns
        -------
        float
            Integrand, shown above.

        """
        x = k * rES * np.cos(theta_kS)

        prefactor = np.power(
            -k * rES * np.sin(theta_kS) * np.tan(theta_kS) / 2.0, i
        ) / np.math.factorial(i)

        return prefactor * (
            (2 + i) * besselJ(3.0 / 2 + i, x) - x * besselJ(5.0 / 2 + i, x)
        )

    # /def

    def angular_sum(
        self, rES: float, k: float, theta_kS: float, *, i_max: int = 100,
    ):
        r"""Angular Summation over Angular Summand.

        .. math::

            2\left(\frac{2\pi}{k r_{SE}\cos\theta_{kS}}\right)^{3/2}
            \sum_{m=0}^\infty \frac{-1^m}{2^{m}m!} \left(k r_{SE} \sin\theta_{kS}\tan\theta_{kS}\right)^m
            \left[(2+m) J_{\frac{3}{2}+m}(k r_{SE} \cos\theta_{kS}) - k r_{SE} \cos\theta_{kS} J_{\frac{5}{2}+m}(k r_{SE} \cos\theta_{kS}) \right]

        Parameters
        ----------
        rES : float
            Distance from :math:`\zeta_E` to :math:`zeta_S`, [Mpc].

        k : float
            :math:`k` magnitude, [1/Mpc].
        theta_kS : float
            Angle between :math:`k` and scattering location, [radians].

        Returns
        -------
        float
            Integrand, shown above.

        Other Parameters
        ----------------
        i_max : int, optional, keyword only
            Maximum index in summation

        """
        # 2\left(\frac{2\pi}{k r_{SE}\cos\theta_{kS}}\right)^{3/2}
        prefactor = 2 * np.power(
            2 * np.pi / (k * rES * np.cos(theta_kS)), 3.0 / 2
        )

        # summation indices
        ms = np.arange(0, i_max, 1)

        return prefactor * np.sum(
            self.angular_summand(ms, rES=rES, k=k, theta_kS=theta_kS)
        )

    # /def

    # ------------------------------

    def _emission_integrand(
        self,
        zetaE: float,
        zetaS: float,
        k: float,
        theta_kS: float,
        i_max: int = 100,
    ):
        r"""Emission Integrand.

        .. math::

            \!\!\left.\int_{\zeta_S}^{\infty}\right\vert_{\Omega_{SE}}\!\!\!\!\!\!\!
                \rm{d}{\zeta_E} \frac{\bar{g}_\gamma^{CL}(\zeta_E)}
                                     {\sqrt{(1+\zeta_E)\zeta_E^3}}
            2\left(\frac{2\pi}{k r_{SE}\cos\theta_{kS}}\right)^{3/2}

            \sum_{m=0}^\infty \frac{-1^m}{2^{m}m!} \left(k r_{SE} \sin\theta_{kS}\tan\theta_{kS}\right)^m
            \left[(2+m) J_{\frac{3}{2}+m}(k r_{SE} \cos\theta_{kS}) - k r_{SE} \cos\theta_{kS} J_{\frac{5}{2}+m}(k r_{SE} \cos\theta_{kS}) \right]

        Parameters
        ----------
        zetaE, zetaS : float
            :math:`\zeta` at emission and scatter, respectively.

        k : float
            :math:`k` magnitude, [1/Mpc].
        theta_kS : float
            Angle between :math:`k` and scattering location, [radians].

        Returns
        -------
        float
            Integrand, shown above.

        Other Parameters
        ----------------
        i_max : int, optional
            Maximum index in summation

        """
        # calculate rES
        rES: float = self._rMag_Mpc(zetaE, zetaS)

        # return integrand
        emission_integrand = (
            self.GgamBarCL(zetaE)
            * self.angular_sum(rES=rES, k=k, theta_kS=theta_kS, i_max=i_max)
            / np.sqrt((1.0 + zetaE) * zetaE ** 3)
        )

        return emission_integrand

    # /def

    def emission_integral(
        self,
        zetaS: float,
        k: np.ndarray,
        theta_kS: float,
        *,
        zeta_max: float = np.inf,
        i_max: int = 100,
        **integration_kwargs,
    ):
        r"""Emission Integral.

        .. math::

            \!\!\left.\int_{\zeta_S}^{\infty}\right\vert_{\Omega_{SE}}\!\!\!\!\!\!\!
                \rm{d}{\zeta_E} \frac{\bar{g}_\gamma^{CL}(\zeta_E)}
                                     {\sqrt{(1+\zeta_E)\zeta_E^3}}
            2\left(\frac{2\pi}{k r_{SE}\cos\theta_{kS}}\right)^{3/2}

            \sum_{m=0}^\infty \frac{-1^m}{2^{m}m!} \left(k r_{SE} \sin\theta_{kS}\tan\theta_{kS}\right)^m
            \left[(2+m) J_{\frac{3}{2}+m}(k r_{SE} \cos\theta_{kS}) - k r_{SE} \cos\theta_{kS} J_{\frac{5}{2}+m}(k r_{SE} \cos\theta_{kS}) \right]

        Parameters
        ----------
        zetaE, zetaS : float
            :math:`\zeta` at emission and scatter, respectively.

        k : float
            :math:`k` magnitude, [1/Mpc].
        theta_kS : float
            Angle between :math:`k` and scattering location, [radians].

        Returns
        -------
        ingegral : float
            Result of integrating ``emission_integrand``
        residual : float
            Integration residual.

        Other Parameters
        ----------------
        zeta_max : float, optional, keyword only
            Maximum :math:`\zeta`
        i_max : int, optional, keyword only
            Maximum index in summation

        """
        # integrate emission integrand
        integral, residual = self.integration_method(
            self._emission_integrand,
            zetaS,
            zeta_max,
            args=(zetaS, k, theta_kS, i_max),
            **integration_kwargs,
        )
        return integral, residual

    # /def

    # ------------------------------

    def _scatter_integrand(
        self,
        zetaS: float,
        k: float,
        theta_kS: float,
        zeta_max: float = np.inf,
        i_max: int = 100,
        **integration_kwargs,
    ):
        r"""Scatter Integrand.

        .. math::

            \!\!\int_{\zeta_O}^{\infty} \!\!\!\! \rm{d}{\zeta_S}
                \frac{\bar{g}_\gamma^{CL}(\zeta_S) e^{ikr_{S}\hat{k}\cdot\hat{z}}}
                     {\bar{P}_{\gamma}^{CL}(\zeta_S)\sqrt{(1+\zeta_S)\zeta_S^3}}
            \!\!\left.\int_{\zeta_S}^{\infty}\right\vert_{\Omega_{SE}}\!\!\!\!\!\!\!
                \rm{d}{\zeta_E} \frac{\bar{g}_\gamma^{CL}(\zeta_E)}
                                     {\sqrt{(1+\zeta_E)\zeta_E^3}}
            2\left(\frac{2\pi}{k r_{SE}\cos\theta_{kS}}\right)^{3/2}

            \sum_{m=0}^\infty \frac{-1^m}{2^{m}m!} \left(k r_{SE} \sin\theta_{kS}\tan\theta_{kS}\right)^m
            \left[(2+m) J_{\frac{3}{2}+m}(k r_{SE} \cos\theta_{kS}) - k r_{SE} \cos\theta_{kS} J_{\frac{5}{2}+m}(k r_{SE} \cos\theta_{kS}) \right]

        .. todo::

            Check integration bounds
            Pass integration kwargs down

        Parameters
        ----------
        zetaS : float
            :math:`\zeta` at location of scatter.

        k : float
            :math:`k` magnitude, [1/Mpc].
        theta_kS : float
            Angle between :math:`k` and scattering location, [radians].

        Returns
        -------
        float

        Other Parameters
        ----------------
        zeta_max : float, optional
            Maximum :math:`\zeta`
        i_max : int, optional
            Maximum index in summation

        """
        kdotzhat = k * np.cos(theta_kS)

        rS: float = self._rMag_Mpc(zetaS, self.zeta0)
        g_ratio = self.GgamBarCL(zetaS) / self.PgamBarCL(zetaS)

        integral, _ = self.emission_integral(
            zetaS,
            k=k,
            theta_kS=theta_kS,
            zeta_max=zeta_max,
            i_max=i_max,
            **integration_kwargs,
        )

        return (
            g_ratio
            * np.exp(1j * rS * kdotzhat)
            / np.sqrt((1.0 + zetaS) * zetaS ** 3)
            * integral
        )

    # /def

    def scatter_integral(
        self,
        k: float,
        theta_kS: float,
        *,
        zeta_max: float = np.inf,
        i_max: int = 100,
        **integration_kwargs,
    ):
        r"""Scatter Integrand.

        .. math::

            \!\!\int_{\zeta_O}^{\infty} \!\!\!\! \rm{d}{\zeta_S}
                \frac{\bar{g}_\gamma^{CL}(\zeta_S) e^{ikr_{S}\hat{k}\cdot\hat{z}}}
                     {\bar{P}_{\gamma}^{CL}(\zeta_S)\sqrt{(1+\zeta_S)\zeta_S^3}}
            \!\!\left.\int_{\zeta_S}^{\infty}\right\vert_{\Omega_{SE}}\!\!\!\!\!\!\!
                \rm{d}{\zeta_E} \frac{\bar{g}_\gamma^{CL}(\zeta_E)}
                                     {\sqrt{(1+\zeta_E)\zeta_E^3}}
            2\left(\frac{2\pi}{k r_{SE}\cos\theta_{kS}}\right)^{3/2}

            \sum_{m=0}^\infty \frac{-1^m}{2^{m}m!} \left(k r_{SE} \sin\theta_{kS}\tan\theta_{kS}\right)^m
            \left[(2+m) J_{\frac{3}{2}+m}(k r_{SE} \cos\theta_{kS}) - k r_{SE} \cos\theta_{kS} J_{\frac{5}{2}+m}(k r_{SE} \cos\theta_{kS}) \right]

        .. todo::

            Check integration bounds
            Pass integration kwargs down

        Parameters
        ----------
        k : float
            :math:`k` magnitude, [1/Mpc].
        theta_kS : float
            Angle between :math:`k` and scattering location, [radians].

        Returns
        -------
        ingegral : float
            Result of integrating ``scatter_integrand``
        residual : float
            Integration residual.

        Other Parameters
        ----------------
        zeta_max : float, optional, keyword only
            Maximum :math:`\zeta`
        i_max : int, optional, keyword only
            Maximum index in summation.

        """
        # integrate scatter integrand
        integral, residual = self.integration_method(
            self._scatter_integrand,
            # bounds
            self.zeta0,
            zeta_max,
            # arguments
            args=(k, theta_kS, zeta_max, i_max),
            **integration_kwargs,
        )
        return integral, residual

    # /def

    # ------------------------------

    @u.quantity_input(freq=u.GHz, k=1 / u.Mpc, theta_kS=u.rad)
    def __call__(
        self,
        freq: QuantityType,
        k: QuantityType,
        theta_kS: QuantityType,
        *,
        zeta_max: float = np.inf,
        i_max: int = 100,
        **integration_kwargs,
    ):
        r"""Perform Calculation.

        .. math::

            \frac{I^{(sd)}(\nu,\hat{n})}{B_\nu(\nu, T_0)} =
            \left(
            \frac{\lambda_0^2 A(\vec{k})}{16\pi \bar{P}_{\gamma}^{CL}(\zeta_0)}
            \frac{h\nu/k_B T_0}{e^{-\frac{h\nu}{k_B T_0}}-1}
            \!\!\int_{\zeta_O}^{\infty} \!\!\!\! \rm{d}{\zeta_S}
                \frac{\bar{g}_\gamma^{CL}(\zeta_S) e^{ikr_{S}\hat{k}\cdot\hat{z}}}
                     {\bar{P}_{\gamma}^{CL}(\zeta_S)\sqrt{(1+\zeta_S)\zeta_S^3}}
            \!\!\left.\int_{\zeta_S}^{\infty}\right\vert_{\Omega_{SE}}\!\!\!\!\!\!\!
                \rm{d}{\zeta_E} \frac{\bar{g}_\gamma^{CL}(\zeta_E)}
                                     {\sqrt{(1+\zeta_E)\zeta_E^3}}
            2\left(\frac{2\pi}{k r_{SE}\cos\theta_{kS}}\right)^{3/2}
            \right.

            \left.
            \sum_{m=0}^\infty \frac{-1^m}{2^{m}m!} \left(k r_{SE} \sin\theta_{kS}\tan\theta_{kS}\right)^m
            \left[(2+m) J_{\frac{3}{2}+m}(k r_{SE} \cos\theta_{kS}) - k r_{SE} \cos\theta_{kS} J_{\frac{5}{2}+m}(k r_{SE} \cos\theta_{kS}) \right]
            \right)

        .. todo::

            Check integration bounds
            Pass integration kwargs down

        Parameters
        ----------
        freq : `~astropy.units.Quantity`
            Frequency [GHz].
        k : float
            :math:`k` magnitude, [1/Mpc].
        theta_kS : float
            Angle between :math:`k` and scattering location, [radians].

        Returns
        -------
        `~astropy.units.Quantity`

        Other Parameters
        ----------------
        zeta_max : float, optional, keyword only
            Maximum :math:`\zeta`
        i_max : int, optional, keyword only
            Maximum index in summation.

        """
        # integrate scatter integrand
        integral, _ = self.scatter_integral(
            k.to(1 / u.Mpc).value,  # ensure value copy
            theta_kS.to(u.rad).value,  # ensure value copy
            zeta_max=zeta_max,
            i_max=i_max,
            **integration_kwargs,
        )

        return self.prefactor(freq=freq, k=k) * integral

    compute = __call__  # alias
    # /def

    #######################################################

    def plot_PgamBarCL(self, plot_times: bool = False):
        import matplotlib.pyplot as plt

        plt.scatter(
            self._zeta_arr,
            self.PgamBarCL(self._zeta_arr),
            label=r"$\bar{P}_{\gamma}^{\mathrm{CL}}(\zeta)$",
        )

        # some interesting times
        if plot_times:
            plt.axvline(self.zeta0, label=r"$\zeta_0$", c="k", ls="-")
            plt.axvline(1, label=r"$\zeta_{\mathrm{eq}}$", c="k", ls="--")
            plt.axvline(self.zeta(1100), label=r"$\zeta_{R}$", c="k", ls=":")

        plt.legend()

        # return fig

    # /def

    def plot_GgamBarCL(self, plot_times: bool = False):
        import matplotlib.pyplot as plt

        plt.scatter(
            self._zeta_arr,
            self.GgamBarCL(self._zeta_arr),
            label=r"$\bar{P}_{\gamma}^{\mathrm{CL}}(\zeta)$",
        )

        # some interesting times
        if plot_times:
            plt.axvline(self.zeta0, label=r"$\zeta_0$", c="k", ls="-")
            plt.axvline(1, label=r"$\zeta_{\mathrm{eq}}$", c="k", ls="--")
            plt.axvline(self.zeta(1100), label=r"$\zeta_{R}$", c="k", ls=":")

        plt.legend()

        # return fig

    # /def


# /class

##############################################################################
# END
