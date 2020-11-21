# -*- coding: utf-8 -*-

"""Gray Body."""

__all__ = [
    "GrayBody",
]


##############################################################################
# IMPORTS

import typing as T

import astropy.units as u
import numpy as np
from scipy.special import jv as besselJ

from .core import IntrinsicDistortionBase, ArrayLike_Callable

##############################################################################
# PARAMETERS

QuantityType = T.TypeVar("Quantity", u.Quantity, u.SpecificTypeQuantity)


##############################################################################
# CODE
##############################################################################


class GrayBody(IntrinsicDistortionBase):
    r"""Gray-body piece of Intrinsic-Y calculation.

    .. todo::

        this function

    ..
      RST SUBSTITUTIONS

    .. |NDarray| replace:: `~numpy.ndarray`
    .. |Quantity| replace:: `~astropy.units.Quantity`

    """

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
        # reduced_energy = freq * const.h / (const.k_B * self.Tcmb0) << u.one

        Ak = self.AkFunc(k)
        if real_AK is True:
            Ak = np.real(Ak)
        elif real_AK is False:
            Ak = np.imaginary(Ak)

        return 2 * self.lambda0 * Ak / self.PgamBarCL0

    # /def

    # ------------------------------

    def scatter_integral(
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
        r"""Scatter integral.

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
        integral : float
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
        graybody : |Quantity|

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
            1.0
            + (r_prefact * np.real(res) + 1j * i_prefact * np.imaginary(res))
        )

    compute = __call__  # alias
    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
