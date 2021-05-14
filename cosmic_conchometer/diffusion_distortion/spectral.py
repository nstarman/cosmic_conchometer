# -*- coding: utf-8 -*-

"""Spectral Distortion."""

__all__ = ["SpectralDistortion"]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.cosmology.core import Cosmology
from classy import Class

# PROJECT-SPECIFIC
from .core import DiffusionDistortionBase, IUSType
from cosmic_conchometer.typing import ArrayLike, TArrayLike, ArrayLike_Callable

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
        class_cosmo: Class,
        *,
        AkFunc: T.Union[str, ArrayLike_Callable, None] = None,
    ) -> None:
        super().__init__(
            cosmo=cosmo,
            class_cosmo=class_cosmo,
            AkFunc=AkFunc,
        )

        # Read in script-C cubes
        raise NotImplementedError("TODO!")

        self._betaDeltas
        self._script_C_nogam
        self._script_C_gamma

        # interpolate them ?

    # /def

    # ------------------------------

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

    # ------------------------------------------------------------

    def _l_sum(self) -> np.ndarray:  # TODO! descriptive name
        raise NotImplementedError("TODO!")

    def _i_sum(self) -> np.ndarray:  # TODO! descriptive name
        raise NotImplementedError("TODO!")

    # ...

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
