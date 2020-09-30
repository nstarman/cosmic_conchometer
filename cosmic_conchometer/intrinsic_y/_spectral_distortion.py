# -*- coding: utf-8 -*-

"""Spectral Distortion."""

__all__ = [
    "SpectralDistortion",
    "SpectralDistortionApproximation"
]


##############################################################################
# IMPORTS

import typing as T
import warnings

import astropy.constants as const
import astropy.units as u
import mpmath
import numpy as np
import scipy.integrate as integ
from astropy.cosmology.core import Cosmology
from scipy.special import spherical_jn as besselJ

from .core import ArrayLike_Callable, IntrinsicDistortionBase

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

        I^{(sd)}(\nu,\hat{n}) =
        {B_\nu(\nu, T_0)}
        \left(
        \frac{\lambda_0^2 A(\vec{k})}{16\pi \bar{P}_{\gamma}^{CL}(\zeta_0)}
        \frac{h\nu/k_B T_0}{e^{-\frac{h\nu}{k_B T_0}}-1}
        \!\!\int_{\zeta_O}^{\infty} \!\!\!\! \rm{d}{\zeta_S}
            \frac{\bar{g}_\gamma^{CL}(\zeta_S) e^{ikr_{S}\hat{k}\cdot\hat{z}}}
                 {\bar{P}_{\gamma}^{CL}(\zeta_S)\sqrt{(1+\zeta_S)\zeta_S^3}}
        \!\!\left.\int_{\zeta_S}^{\infty}\right\vert_{\Omega_{SE}}\!\!\!\!\!\!\!
            \rm{d}{\zeta_E} \frac{\bar{g}_\gamma^{CL}(\zeta_E)}
                                 {\sqrt{(1+\zeta_E)\zeta_E^3}}
        \right.

        (-8\pi^2) \sum_{m=0}^\infty
            \frac{-1^m \tan^{2m}\theta_{kS}}{2^{m}(2m+3)m!}
            \left(k r_{SE} \cos\theta_{kS}\right)^m
            \left[
                (m\!+\!1)~  j_{5/2+m}(kr_{SE}\cos\theta_{kS})
                + (m\!+\!2)~  j_{1/2+m}(kr_{SE}\cos\theta_{kS})
            \right]
        \Bigg{)}

    Parameters
    ----------
    cosmo : :class:`~astropy.cosmology.core.Cosmology`
    class_cosmo : :class:`~classy.Class`
    GgamBarCL : Callable
    PgamBarCL : Callable
    AkFunc: Callable or str or None, optional
    integration_method : Callable

    ..
      RST SUBSTITUTIONS

    .. |NDarray| replace:: :class:`~numpy.ndarray`
    .. |Quantity| replace:: :class:`~astropy.units.Quantity`

    """

    def __init__(
        self,
        cosmo: Cosmology,
        class_cosmo,
        *,
        AkFunc: T.Union[str, ArrayLike_Callable, None] = None,
        integration_method: T.Callable = integ.quad,
    ):
        super().__init__(
            cosmo=cosmo,
            class_cosmo=class_cosmo,
            AkFunc=AkFunc,
            integration_method=integration_method,
        )

        # vectorized angular_integrand
        self.angular_integrand = np.vectorize(self._angular_integrand)

    # /def

    # ------------------------------

    @u.quantity_input(freq=u.GHz, k=1 / u.Mpc)
    def prefactor(
        self,
        freq: QuantityType,
        k: QuantityType,
        real_AK: T.Optional[bool] = None,
    ):
        r"""Combined Prefactor.

        .. math::

            \frac{\lambda_0^2 A(\vec{k})}{16\pi \bar{P}_{\gamma}^{CL}(\zeta_0)}
            \frac{h\nu/k_B T_0}{e^{-\frac{h\nu}{k_B T_0}}-1}

        Parameters
        ----------
        freq : |Quantity|
            The frequency.
        k : |Quantity|
            Units of inverse Mpc.

        Returns
        -------
        |Quantity|
            units of [Mpc**2 * units[AkFunc]]

        Other Parameters
        ----------------
        real_AK : bool or None, optional
            Whether to use the real (True) or imaginary (False) component of
            Ak or return the whole complex factor (None, default)

        """
        reduced_energy = freq * const.h / (const.k_B * self.Tcmb0) << u.one

        Ak = self.AkFunc(k)
        if real_AK is True:
            Ak = np.real(Ak)
        elif real_AK is False:
            Ak = np.imaginary(Ak)

        prefactor = (
            (reduced_energy / np.expm1(-reduced_energy))
            * self.lambda0 ** 2
            * Ak
            / (16 * np.pi * self.PgamBarCL0)
        )

        return prefactor

    # /def

    # ------------------------------

    # @np.vectorize  # TODO implement here
    @staticmethod
    def _angular_integrand(
        m: int, rES: float, k: float, theta_kS: float,
    ):
        r"""Angular Summation.

        .. math::

            \frac{-1^m \tan^{2m}\theta_{kS}}{2^{m}(2m+3)m!}
            \left(k r_{SE} \cos\theta_{kS}\right)^m
            \left[
                (m\!+\!1)~  j_{5/2+m}(kr_{SE}\cos\theta_{kS})
                + (m\!+\!2)~  j_{1/2+m}(kr_{SE}\cos\theta_{kS})
            \right]

        Parameters
        ----------
        m: int
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

        Note
        ----
        this method is vectorized with `~numpy.vectorize`

        """
        x = k * rES * np.cos(theta_kS)

        # \frac{-1^m \tan^{2m}\theta_{kS}}{2^{m}(2m+3)m!}
        prefactor = np.divide(
            np.power(-0.5 * np.tan(theta_kS) ** 2, m),
            (2 * m + 3) * np.factorial(m),
        )

        #  \left(k r_{SE} \cos\theta_{kS}\right)^m \\
        #  [  (m+1) j_{5/2+m}(kr_{SE}\cos\theta_{kS})
        #   + (m+2) j_{1/2+m}(kr_{SE}\cos\theta_{kS}) ]
        return (
            prefactor
            * np.power(x, m)
            * ((m + 1) * besselJ(2.5 + m, x) + besselJ(0.5 + m, x))
        )

    # /def

    def angular_integral(
        self, rES: float, k: float, theta_kS: float, *, m_max: int = 100,
    ):
        r"""Angular Summation over Angular Summand.

        .. math::

            -8\pi^2 \sum_{m=0}^\infty
            \frac{-1^m \tan^{2m}\theta_{kS}}{2^{m}(2m+3)m!}
            \left(k r_{SE} \cos\theta_{kS}\right)^m
            \left[
                (m\!+\!1)~  j_{5/2+m}(kr_{SE}\cos\theta_{kS})
                + (m\!+\!2)~  j_{1/2+m}(kr_{SE}\cos\theta_{kS})
            \right]

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
        m_max : int, optional, keyword only
            Maximum index in summation

        """
        prefactor: float = -8 * np.pi ** 2

        # summation indices
        ms = np.arange(0, m_max, 1)

        return prefactor * np.sum(
            self.angular_integrand(ms, rES=rES, k=k, theta_kS=theta_kS)
        )

    # /def

    # ------------------------------

    def _emission_integrand(
        self,
        zetaE: float,
        zetaS: float,
        k: float,
        theta_kS: float,
        m_max: int = 100,
    ):
        r"""Emission Integrand.

        .. math::

            \!\!\left.\int_{\zeta_S}^{\infty}\right\vert_{\Omega_{SE}}\!\!\!\!\!\!\!
                \rm{d}{\zeta_E} \frac{\bar{g}_\gamma^{CL}(\zeta_E)}
                                     {\sqrt{(1+\zeta_E)\zeta_E^3}}
            \mathrm{angular_integral}(\zeta_E, \zeta_S, |k|, \cos{\theta_{kS}})

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
        m_max : int, optional
            Maximum index in summation

        """
        # calculate rES
        rES: float = self._rMag_Mpc(zetaE, zetaS)

        # return integrand
        emission_integrand = (
            self.GgamBarCL(zetaE)
            * self.angular_integral(
                rES=rES, k=k, theta_kS=theta_kS, m_max=m_max
            )
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
        m_max: int = 100,
        **integration_kwargs,
    ):
        r"""Emission Integral.

        .. math::

            \!\!\left.\int_{\zeta_S}^{\infty}\right\vert_{\Omega_{SE}}\!\!\!\!\!\!\!
                \rm{d}{\zeta_E} \frac{\bar{g}_\gamma^{CL}(\zeta_E)}
                                     {\sqrt{(1+\zeta_E)\zeta_E^3}}
            \ \text{angular sum}(\zeta_E, \zeta_S, |k|, \cos{\theta_{kS}})

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
        integral : float
            Result of integrating ``emission_integrand``
        residual : float
            Integration residual.

        Other Parameters
        ----------------
        zeta_max : float, optional, keyword only
            Maximum :math:`\zeta`
        m_max : int, optional, keyword only
            Maximum index in summation

        """
        # integrate emission integrand
        integral, residual = self.integration_method(
            self._emission_integrand,
            zetaS,
            zeta_max,
            args=(zetaS, k, theta_kS, m_max),
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
        real: T.Optional[bool] = None,
        zeta_max: float = np.inf,
        m_max: int = 100,
        **integration_kwargs,
    ):
        r"""Scatter Integrand.

        .. math::

            \!\!\int_{\zeta_O}^{\infty} \!\!\!\! \rm{d}{\zeta_S}
              \frac{\bar{g}_\gamma^{CL}(\zeta_S) e^{ikr_S\hat{k}\cdot\hat{z}}}
                   {\bar{P}_{\gamma}^{CL}(\zeta_S)\sqrt{(1+\zeta_S)\zeta_S^3}}
            \!\!\left.\int_{\zeta_S}^{\infty}\right\vert_{\Omega_{SE}}\!\!\!\!\!\!\!
                \rm{d}{\zeta_E} \frac{\bar{g}_\gamma^{CL}(\zeta_E)}
                                     {\sqrt{(1+\zeta_E)\zeta_E^3}}
            \ \text{angular sum}(\zeta_E, \zeta_S, |k|, \cos{\theta_{kS}})

        .. todo::

            change from g_ratio to spline on that quantity?

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
        real : None or bool, optional
            Whether to return Real (True) or Imaginary (False)
            or Complex (None, default).
        zeta_max : float, optional
            Maximum :math:`\zeta`
        m_max : int, optional
            Maximum index in summation

        """
        kdotzhat: float = k * np.cos(theta_kS)

        rS: float = self._rMag_Mpc(zetaS, self.zeta0)
        g_ratio = self.GgamBarCL(zetaS) / self.PgamBarCL(zetaS)

        integral, _ = self.emission_integral(
            zetaS,
            k=k,
            theta_kS=theta_kS,
            zeta_max=zeta_max,
            m_max=m_max,
            **integration_kwargs,
        )

        if real is None:
            expRK = np.exp(1j * rS * kdotzhat)
        elif real:  # is True
            expRK = np.cos(rS * kdotzhat)
        else:
            expRK = np.sin(rS * kdotzhat)

        return g_ratio * expRK / np.sqrt((1.0 + zetaS) * zetaS ** 3) * integral

    # /def

    def scatter_integral(
        self,
        k: float,
        theta_kS: float,
        *,
        zeta_min: float = 0.0,
        zeta_max: float = np.inf,
        m_max: int = 100,
        **integration_kwargs,
    ):
        r"""Scatter integral.

        .. math::

            \!\!\int_{\zeta_O}^{\infty} \!\!\!\! \rm{d}{\zeta_S}
              \frac{\bar{g}_\gamma^{CL}(\zeta_S) e^{ikr_S \hat{k}\cdot\hat{z}}}
                   {\bar{P}_{\gamma}^{CL}(\zeta_S)\sqrt{(1+\zeta_S)\zeta_S^3}}
            \text{emission integral}
            \left(
                \text{angular sum}(\zeta_E, \zeta_S, |k|, \cos{\theta_{kS}})
            \right)

        .. todo::

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
        zeta_min, zeta_max : float, optional, keyword only
            Minimum / maximum :math:`\zeta`
        m_max : int, optional, keyword only
            Maximum index in summation.

        """
        # integrate scatter integrand
        real_integral, real_residual = self.integration_method(
            self._scatter_integrand,
            # bounds
            zeta_min,
            zeta_max,
            # arguments
            args=(k, theta_kS, True, zeta_max, m_max),
            **integration_kwargs,
        )

        # integrate scatter integrand
        imag_integral, imag_residual = self.integration_method(
            self._scatter_integrand,
            # bounds
            zeta_min,
            zeta_max,
            # arguments
            args=(k, theta_kS, False, zeta_max, m_max),
            **integration_kwargs,
        )

        return (
            real_integral + 1j * imag_integral,
            (real_residual, imag_residual),
        )

    # /def

    # ------------------------------

    @u.quantity_input(freq=u.GHz, k=1 / u.Mpc, theta_kS=u.rad)
    def __call__(
        self,
        freq: QuantityType,
        k: QuantityType,
        theta_kS: QuantityType,
        *,
        zeta_min: float = 0.0,
        zeta_max: float = np.inf,
        m_max: int = 100,
        **integration_kwargs,
    ):
        r"""Perform Calculation.

        .. math::

            I^{(sd)}(\nu,\hat{n}) =
            {B_\nu(\nu, T_0)}
            \left(
            \frac{\lambda_0^2 A(\vec{k})}{16\pi \bar{P}_{\gamma}^{CL}(\zeta_0)}
            \frac{h\nu/k_B T_0}{e^{-\frac{h\nu}{k_B T_0}}-1}
            \!\!\int_{\zeta_O}^{\infty} \!\!\!\! \rm{d}{\zeta_S}
                \frac{\bar{g}_\gamma^{CL}(\zeta_S) e^{ikr_S\hat{k}\cdot\hat{z}}}
                     {\bar{P}_{\gamma}^{CL}(\zeta_S)\sqrt{(1+\zeta_S)\zeta_S^3}}
            \!\!\left.\int_{\zeta_S}^{\infty}\right\vert_{\Omega_{SE}}\!\!\!\!\!\!\!
                \rm{d}{\zeta_E} \frac{\bar{g}_\gamma^{CL}(\zeta_E)}
                                     {\sqrt{(1+\zeta_E)\zeta_E^3}}
            \right.

            (-8\pi^2) \sum_{m=0}^\infty
            \frac{-1^m \tan^{2m}\theta_{kS}}{2^{m}(2m+3)m!}
            \left(k r_{SE} \cos\theta_{kS}\right)^m
            \left[
                (m\!+\!1)~  j_{5/2+m}(kr_{SE}\cos\theta_{kS})
                + (m\!+\!2)~  j_{1/2+m}(kr_{SE}\cos\theta_{kS})
            \right]
            \Bigg{)}

        The cross-terms in :math:`A(k)` with the integral cancel out.
        Only the real-with-real and imaginary-with-imaginary remain.

        ..
          RST SUBSTITUTIONS

        .. |Quantity| replace:: `~astropy.units.Quantity`

        Parameters
        ----------
        freq : |Quantity|
            Frequency [GHz].
        k : |Quantity|
            :math:`k` magnitude, [1/Mpc].
        theta_kS : |Quantity|
            Angle between :math:`k` and scattering location, [radians].

        Returns
        -------
        spectral_distortion : |Quantity|

        Other Parameters
        ----------------
        zeta_min, zeta_max : float, optional, keyword only
            Minimum / Maximum :math:`\zeta`
        m_max : int, optional, keyword only
            Maximum index in summation.

        """
        # integrate scatter integrand (complex number)
        res, _ = self.scatter_integral(
            k.to(1 / u.Mpc).value,  # ensure value copy
            theta_kS.to(u.rad).value,  # ensure value copy
            zeta_max=zeta_max,
            m_max=m_max,
            **integration_kwargs,
        )

        # prefactor
        r_prefact = self.prefactor(freq=freq, k=k, real_AK=True)
        i_prefact = self.prefactor(freq=freq, k=k, real_AK=False)

        # blackbody
        bb = self.blackbody(freq=freq, temp=self.Tcmb0)

        # correctly multiply the prefactor and integral
        return bb * (
            r_prefact * np.real(res) + 1j * i_prefact * np.imaginary(res)
        )

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


#####################################################################


class SpectralDistortionApproximation(SpectralDistortion):
    r"""Spectral Distortion.

    Parameters
    ----------
    cosmo : :class:`~astropy.cosmology.core.Cosmology`
    class_cosmo : :class:`~classy.Class`
    GgamBarCL : Callable
    PgamBarCL : Callable
    AkFunc: Callable or str or None, optional
    integration_method : Callable

    ..
      RST SUBSTITUTIONS

    .. |NDarray| replace:: :class:`~numpy.ndarray`
    .. |Quantity| replace:: :class:`~astropy.units.Quantity`

    """

    def __init__(
        self,
        cosmo: Cosmology,
        class_cosmo,
        *,
        AkFunc: T.Union[str, ArrayLike_Callable, None] = None,
        integration_method: T.Callable = integ.quad,
    ):
        super().__init__(
            cosmo=cosmo,
            class_cosmo=class_cosmo,
            AkFunc=AkFunc,
            integration_method=integration_method,
        )

        # Get the spline information
        warnings.warn("TODO: Need to do Spline Knots")

        self.GgamBarCL_coeffs = self.GgamBarCL.get_coeffs()
        self.PgamBarCL_coeffs = self.PgamBarCL.get_coeffs()

    # /def

    # ------------------------------

    @u.quantity_input(freq=u.GHz, k=1 / u.Mpc)
    def prefactor(
        self,
        freq: QuantityType,
        k: QuantityType,
        real_AK: T.Optional[bool] = None,
    ):
        r"""Combined Prefactor.

        .. math::

            \frac{\pi \lambda_0 A(\vec{k})}{\bar{P}_{\gamma}^{CL}(\zeta_0)}
            \frac{h\nu/k_B T_0}{e^{-\frac{h\nu}{k_B T_0}}-1}

        Parameters
        ----------
        freq : |Quantity|
            The frequency.
        k : |Quantity|
            Units of inverse Mpc.

        Returns
        -------
        |Quantity|
            units of [Mpc**2 * units[AkFunc]]

        Other Parameters
        ----------------
        real_AK : bool or None, optional
            Whether to use the real (True) or imaginary (False) component of
            Ak or return the whole complex factor (None, default)

        """
        reduced_energy = freq * const.h / (const.k_B * self.Tcmb0) << u.one

        Ak = self.AkFunc(k)
        if real_AK is True:
            Ak = np.real(Ak)
        elif real_AK is False:
            Ak = np.imaginary(Ak)

        prefactor = (
            (reduced_energy / np.expm1(-reduced_energy))
            * np.pi
            * self.lambda0
            * Ak
            / self.PgamBarCL0
        )

        return prefactor

    # /def

    # ------------------------------

    def _emission_integrand(
        self,
        n: int,  # emission constant index
        m: int,  # angular summand index
        i: int,  # zetaE index
        zetaS: float,
        k: float,
        theta_kS: float,
    ):
        r"""Emission Integrand.

        Parameters
        ----------
        n : int
            Emission constant index. Must be in range [0, 6].
        m : int
            Angular summand index.
        i : int
            :math:`\zeta_E` index.
        zetaS : float
            :math:`\zeta_S` at scatter.
        k : float
            :math:`k` magnitude, [1/Mpc].
        theta_kS : float
            Angle between :math:`k` and scattering location, [radians].

        Returns
        -------
        emission_integrand : float

        """
        # ------------------------------
        # 1) Calculate the coefficient
        inv_zS = 1.0 / zetaS

        # spline coefficient
        a0 = self.GgamBarCL_coeffs[0, i]
        a1 = self.GgamBarCL_coeffs[1, i]
        a2 = self.GgamBarCL_coeffs[2, i]
        a3 = self.GgamBarCL_coeffs[3, i]

        if n == 0:
            Cn = a0 + (a1 * inv_zS) + (a2 * inv_zS ** 2) + (a3 * inv_zS ** 3)
        elif n == 1:
            Cn = (
                2
                * np.sqrt(1 + inv_zS)
                * (a1 + 2 * a2 * inv_zS + 3 * a3 * inv_zS ** 2)
            )
        elif n == 2:
            Cn = (
                a1
                + 2 * a2 * (2 + 3 * inv_zS)
                + 3 * a3 * (5 * inv_zS ** 2 + 4 * inv_zS)
            )
        elif n == 3:
            Cn = 4 * np.sqrt(1 + inv_zS) * (a2 + a3 * (2 + 5 * inv_zS))
        elif n == 4:
            Cn = a2 + 3 * a3 * (4 + 5 * inv_zS)
        elif n == 5:
            Cn = 6 * a3
        elif n == 6:
            Cn = a3
        else:
            raise ValueError

        # ------------------------------
        # 2) Calculate the integrals
        jm_coeff: float = (m + 2) / (2 * m + n + 1) / np.factorial(2 * m + 1)
        jmp2_coeff: float = (
            8
            * (m + 3)
            * (m + 2)
            * (m + 1)
            / (2 * m + n + 3)
            / np.factorial(2 * m + 6)
        )

        # ---------------
        # x_i+1
        zetaE_ip1: float = self._zeta_arr[i + 1]
        rES_ip1: float = self._rMag_Mpc(zetaE_ip1, zetaS)
        x_ip1: float = k * rES_ip1 * np.cos(theta_kS)
        # jm - piece
        jm_ip1 = mpmath.hyp1f2(
            m + n / 2 + 0.5, m + 1.5, m + n / 2 + 1.5, -(x_ip1 ** 2) / 4
        )
        # jm+2 - piece
        jmp2_ip1 = mpmath.hyp1f2(
            m + n / 2 + 1.5, m + 3.5, m + n / 2 + 2.5, -(x_ip1 ** 2) / 4
        )
        # put it together
        Fx_ip1 = (jm_coeff * jm_ip1 + jmp2_coeff * jmp2_ip1) * np.power(
            x_ip1, 2 * m + n + 1
        )

        # ---------------
        # x_i
        zetaE_i: float = self._zeta_arr[i]
        rES_i: float = self._rMag_Mpc(zetaE_i, zetaS)
        x_i: float = k * rES_i * np.cos(theta_kS)
        # jm - piece
        jm_i = mpmath.hyp1f2(
            m + n / 2 + 0.5, m + 1.5, m + n / 2 + 1.5, -(x_i ** 2) / 4
        )
        # jm+2 - piece
        jmp2_i = mpmath.hyp1f2(
            m + n / 2 + 1.5, m + 3.5, m + n / 2 + 2.5, -(x_i ** 2) / 4
        )
        # put it together
        Fx_i = (jm_coeff * jm_i + jmp2_coeff * jmp2_i) * np.power(
            x_i, 2 * m + n + 1
        )

        # ---------------
        # return
        x0: float = k * self._lambda0_Mpc * np.cos(theta_kS)
        return Cn * (Fx_ip1 - Fx_i) / x0

    # /def

    def emission_integral(
        self, m: int, i: int, zetaS: float, k: float, theta_kS: float,
    ):
        r"""Emission Integral.

        Parameters
        ----------
        m : int
            Angular summand index.
        i : int
            :math:`\zeta_E` index.
        zetaS : float
            :math:`\zeta_S` at scatter.
        k : float
            :math:`k` magnitude, [1/Mpc].
        theta_kS : float
            Angle between :math:`k` and scattering location, [radians].

        Returns
        -------
        emission_integral : float
            summation over ``_emission_integrand`` for n in [0, 6]

        """
        summation = np.sum(
            (
                self._emission_integrand(
                    n, m=m, i=i, zetaS=zetaS, k=k, theta_kS=theta_kS
                )
                for n in range(6 + 1)  # n_max = 6 (inclusive)
            )
        )

        return summation

    # /def

    # ------------------------------

    def _angular_integrand(
        self, m: int, i: int, zetaS: float, k: float, theta_kS: float,
    ):
        r"""Angular Summand.

        Parameters
        ----------
        m : int
            Angular summand index.
        i : int
            :math:`\zeta_E` index.
        zetaS : float
            :math:`\zeta_S` at scatter.
        k : float
            :math:`k` magnitude, [1/Mpc].
        theta_kS : float
            Angle between :math:`k` and scattering location, [radians].

        Returns
        -------
        angular_integrand : float

        """
        prefactor = np.power(-1 * np.tan(theta_kS) ** 2) / (2 * m + 3)

        return prefactor * self.emission_integral(
            m, i, zetaS=zetaS, k=k, theta_kS=theta_kS
        )

    # /def

    def angular_integral(
        self, zetaS: float, k: float, theta_kS: float, *, m_max: int = 100,
    ):
        r"""Angular Summation.

        .. todo::

            Check on summation bounds

        Parameters
        ----------
        zetaS : float
            :math:`\zeta_S` at scatter.
        k : float
            :math:`k` magnitude, [1/Mpc].
        theta_kS : float
            Angle between :math:`k` and scattering location, [radians].

        Returns
        -------
        angular_integral : float

        Other Parameters
        ----------------
        m_max : int, optional, keyword only
            Maximum index in summation

        """
        summation = np.sum(
            np.sum(
                (
                    self._angular_integrand(
                        m, i=i, zetaS=zetaS, k=k, theta_kS=theta_kS
                    )
                    for m in range(m_max + 1)
                )
            )
            for i in range(len(self._zeta_arr))
        )

        return summation

    # /def

    # ------------------------------

    def _scatter_integrand(
        self,
        zetaS: float,
        k: float,
        theta_kS: float,
        real: T.Optional[bool] = None,
        m_max: int = 100,
    ):
        r"""Scatter Integrand.

        Parameters
        ----------
        zetaS : float
            :math:`\zeta_S` at scatter.
        k : float
            :math:`k` magnitude, [1/Mpc].
        theta_kS : float
            Angle between :math:`k` and scattering location, [radians].

        Returns
        -------
        scatter_integrand : float

        Other Parameters
        ----------------
        real : bool, optional
            Whether to take the real or imaginary part.
        m_max : int, optional
            Maximum index in summation

        """
        kdotzhat: float = k * np.cos(theta_kS)

        rS: float = self._rMag_Mpc(zetaS, self.zeta0)
        g_ratio = self.GgamBarCL(zetaS) / self.PgamBarCL(zetaS)

        integral = self.angular_integral(
            zetaS, k=k, theta_kS=theta_kS, m_max=m_max
        )

        if real is None:
            expRK = np.exp(1j * rS * kdotzhat)
        elif real:  # is True
            expRK = np.cos(rS * kdotzhat)
        else:
            expRK = np.sin(rS * kdotzhat)

        return g_ratio * expRK / np.sqrt((1.0 + zetaS) * zetaS ** 3) * integral

    # /def

    def scatter_integral(
        self,
        k: float,
        theta_kS: float,
        *,
        zeta_min: float = 0.0,
        zeta_max: float = np.inf,
        m_max: int = 100,
        **integration_kwargs,
    ):
        r"""Scatter integral.

        Parameters
        ----------
        k : float
            :math:`k` magnitude, [1/Mpc].
        theta_kS : float
            Angle between :math:`k` and scattering location, [radians].

        Returns
        -------
        scatter_integrand : float

        Other Parameters
        ----------------
        zeta_min, zeta_max : float, optional, keyword only
            Minimum / maximum :math:`\zeta`
        m_max : int, optional, keyword only
            Maximum index in summation
        **integration_kwargs

        """
        # integrate scatter integrand
        real_integral, real_residual = self.integration_method(
            self._scatter_integrand,
            # bounds
            zeta_min,
            zeta_max,
            # arguments
            args=(k, theta_kS, True, zeta_max, m_max),
            **integration_kwargs,
        )

        # integrate scatter integrand
        imag_integral, imag_residual = self.integration_method(
            self._scatter_integrand,
            # bounds
            zeta_min,
            zeta_max,
            # arguments
            args=(k, theta_kS, False, zeta_max, m_max),
            **integration_kwargs,
        )

        return (
            real_integral + 1j * imag_integral,
            (real_residual, imag_residual),
        )

    # /def

    # ------------------------------

    @u.quantity_input(freq=u.GHz, k=1 / u.Mpc, theta_kS=u.rad)
    def __call__(
        self,
        freq: QuantityType,
        k: QuantityType,
        theta_kS: QuantityType,
        *,
        zeta_min: float = 0.0,
        zeta_max: float = np.inf,
        m_max: int = 100,
        **integration_kwargs,
    ):
        r"""Perform Calculation.

        The cross-terms in :math:`A(k)` with the integral cancel out.
        Only the real-with-real and imaginary-with-imaginary remain.

        ..
          RST SUBSTITUTIONS

        .. |Quantity| replace:: `~astropy.units.Quantity`

        Parameters
        ----------
        freq : |Quantity|
            Frequency [GHz].
        k : |Quantity|
            :math:`k` magnitude, [1/Mpc].
        theta_kS : |Quantity|
            Angle between :math:`k` and scattering location, [radians].

        Returns
        -------
        spectral_distortion : |Quantity|

        Other Parameters
        ----------------
        zeta_min, zeta_max : float, optional, keyword only
            Minimum / Maximum :math:`\zeta`
        m_max : int, optional, keyword only
            Maximum index in summation.

        """
        # integrate scatter integrand (complex number)
        res, _ = self.scatter_integral(
            k.to(1 / u.Mpc).value,  # ensure value copy
            theta_kS.to(u.rad).value,  # ensure value copy
            zeta_max=zeta_max,
            m_max=m_max,
            **integration_kwargs,
        )

        # prefactor
        r_prefact = self.prefactor(freq=freq, k=k, real_AK=True)
        i_prefact = self.prefactor(freq=freq, k=k, real_AK=False)

        # blackbody
        bb = self.blackbody(freq=freq, temp=self.Tcmb0)

        # correctly multiply the prefactor and integral
        return bb * (
            r_prefact * np.real(res) + 1j * i_prefact * np.imaginary(res)
        )

    # /def


# /class

##############################################################################
# END
