# -*- coding: utf-8 -*-

"""Spectral Distortion."""

__all__ = ["SpectralDistortion"]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T
import warnings

# THIRD PARTY
import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.cosmology.core import Cosmology
from mpmath import hyp1f2, hyp2f2
from scipy.special import gamma
from scipy.special import spherical_jn as besselJ

# PROJECT-SPECIFIC
from .core import ArrayLike_Callable, IntrinsicDistortionBase

##############################################################################
# PARAMETERS

IUSType = T.Callable[[T.Union[float, np.ndarray]], np.ndarray]
QuantityType = T.TypeVar("Quantity", u.Quantity, u.SpecificTypeQuantity)

##############################################################################
# CODE
##############################################################################


@np.vectorize
def _scriptC_individual(
    betaDelta: float, M: int, m: int, rhoES: int, l: int
) -> float:
    """Eqn 115 of https://www.overleaf.com/project/5efe491b4140390001b1c892

    TODO! there are two copies of this function, the other is in scripts.
    Validate that they are the same with a test on the __code__.

    .. math::

        \frac{1}{\beta\Delta} \Bigg[\tilde{\rho}_{ES}^{M} \Exp{i\beta\Delta\tilde{\rho}_{ES}} j_{m+1}(\beta\Delta \tilde{\rho}_{ES})\Bigg]
        - \frac{\sqrt{\pi } \tilde{\rho}_{ES}^{M+1} (\beta\Delta \tilde{\rho}_{ES})^m}{2^{m+2}  \Gamma \left(\frac{2m\!+\!5}{2}\right)}
        \Bigg[
            \frac{i\beta\Delta \tilde{\rho}_{ES}}{(M\!+\!m\!+\!2)} \, {_2F_2}(m\!+\!2,m\!+\!M\!+\!2;2m\!+\!4,m\!+\!M\!+\!3;2i \beta\Delta \tilde{\rho}_{ES})
            -\frac{2m^2\!+\!6m\!-\!M\!+\!5}{(M\!+\!m\!+\!1)} \, {_2F_2}(m\!+\!2,m\!+\!M\!+\!1;2m\!+\!4,m\!+\!M\!+\!2;2i \beta\Delta \tilde{\rho}_{ES})
        \Bigg]

    Parameters
    ----------
    betaDelta:
        .. math::

            \beta = |\vec{k}| \lambda_0 \cos{\theta_{kS}}
            \Delta = \frac{1}{N} \sqrt(\frac{(1+1/a_{eq})}{2})

    M : int
    m : int
    rhoES : int
    l : int

    Returns
    -------
    float

    """
    x = betaDelta * rhoES
    prefactor = power(rhoES / (l + 1), M)

    t1 = exp(1j * x) * besselJ(m + 1, x) / power(betaDelta, m + 1)
    t2 = sqrt(pi) / power(2, m + 2) * power(rhoES, m + 1) / gamma(m + 2.5)
    t3 = (
        (1j * x)
        / (M + m + 2)
        * hyp2f2(m + 2, m + M + 2, 2 * m + 4, m + M + 3, 2j * x)
    )
    # (2m^2+6m-M+5) / (M+m+1) * hyp2f2(m+2, m+M+1, 2m+4, m+M+2, 2i x)
    t4 = (
        (2 * m ** 2 + 6 * m - M + 5)
        / (M + m + 1)
        * hyp2f2(m + 2, m + M + 1, 2 * m + 4, m + M + 2, 2j * x)
    )

    return prefactor * (t1 - t2 * (t3 - t4))


# /def


@np.vectorize
def _scriptC(bD: float, M: int, m: int, l: int) -> float:
    """Eqn 115 of https://www.overleaf.com/project/5efe491b4140390001b1c892

    .. math::

        \frac{1}{\beta\Delta} \Bigg[\tilde{\rho}_{ES}^{M} \Exp{i\beta\Delta\tilde{\rho}_{ES}} j_{m+1}(\beta\Delta \tilde{\rho}_{ES})\Bigg]
        - \frac{\sqrt{\pi } \tilde{\rho}_{ES}^{M+1} (\beta\Delta \tilde{\rho}_{ES})^m}{2^{m+2}  \Gamma \left(\frac{2m\!+\!5}{2}\right)}
        \Bigg[
            \frac{i\beta\Delta \tilde{\rho}_{ES}}{(M\!+\!m\!+\!2)} \, {_2F_2}(m\!+\!2,m\!+\!M\!+\!2;2m\!+\!4,m\!+\!M\!+\!3;2i \beta\Delta \tilde{\rho}_{ES})
            -\frac{2m^2\!+\!6m\!-\!M\!+\!5}{(M\!+\!m\!+\!1)} \, {_2F_2}(m\!+\!2,m\!+\!M\!+\!1;2m\!+\!4,m\!+\!M\!+\!2;2i \beta\Delta \tilde{\rho}_{ES})
        \Bigg]

    Parameters
    ----------
    betaDelta:
        .. math::

            \beta = |\vec{k}| \lambda_0 \cos{\theta_{kS}}
            \Delta = \frac{1}{N} \sqrt(\frac{(1+1/a_{eq})}{2})

    M : int
    m : int
    l : int

    Returns
    -------
    float

    """

    return +_scriptC_individual(
        bD, M, m, rhoES=l + 1, l=l
    ) - _scriptC_individual(bD, M, m, rhoES=l, l=l)


# /def

##############################################################################


class SpectralDistortion(IntrinsicDistortionBase):
    r"""Spectral Distortion.

    Parameters
    ----------
    cosmo : :class:`~astropy.cosmology.core.Cosmology`
        The cosmology
    class_cosmo : :class:`~classy.Class`
    GgamBarCL : Callable
    PgamBarCL : Callable
    AkFunc: Callable or str or None, optional

    """

    def __init__(
        self,
        cosmo: Cosmology,
        class_cosmo,
        *,
        AkFunc: T.Union[str, ArrayLike_Callable, None] = None,
    ):
        super().__init__(
            cosmo=cosmo,
            class_cosmo=class_cosmo,
            AkFunc=AkFunc,
        )

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

    # ------------------------------------------------------------

    # ===============================================================
    # Convenience Methods

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
        # # integrate scatter integrand (complex number)
        # res, _ = self.scatter_integral(
        #     k.to(1 / u.Mpc).value,  # ensure value copy
        #     theta_kS.to(u.rad).value,  # ensure value copy
        #     zeta_max=zeta_max,
        #     m_max=m_max,
        #     **integration_kwargs,
        # )

    #
    # # prefactor
    # r_prefact = self.prefactor(freq=freq, k=k, real_AK=True)
    # i_prefact = self.prefactor(freq=freq, k=k, real_AK=False)
    #
    # # blackbody
    # bb = self.blackbody(freq=freq, temp=self.Tcmb0)
    #
    # # correctly multiply the prefactor and integral
    # return bb * (
    #     r_prefact * np.real(res) + 1j * i_prefact * np.imaginary(res)
    # )

    compute = __call__
    # /def

    #######################################################

    def plot_PgamBarCL(self, plot_times: bool = False):
        # THIRD PARTY
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
        # THIRD PARTY
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
