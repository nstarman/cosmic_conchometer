# -*- coding: utf-8 -*-

"""**DOCSTRING**.

Description.

"""

__all__ = [
    "SpectralDistortion",
]


##############################################################################
# IMPORTS

# BUILT-IN

import typing as T

# THIRD PARTY

import astropy.constants as const
import astropy.units as u

import numpy as np

import scipy.integrate as integ
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

# PROJECT-SPECIFIC

from cosmic_conchometer import default_Ak

from .core import CosmologyDependent


##############################################################################
# PARAMETERS

_ArrayLike_Callable = T.Callable[
    [T.Union[float, np.ndarray]], T.Union[float, np.ndarray]
]

_IUSType = T.Callable[[T.Union[float, np.ndarray]], np.ndarray]


##############################################################################
# CODE
##############################################################################


class SpectralDistortionBase(CosmologyDependent):
    """Spectral Distortion.

    Parameters
    ----------
    cosmo : `~astropy.cosmology.Cosmology`
    class_cosmo : `~classy.Class`
    AkFunc: Callable or str or None, optional, keyword only

    Other Parameters
    ----------------
    integration_method : Callable

    """

    def __init__(
        self,
        cosmo,
        class_cosmo,
        *,
        AkFunc: T.Union[str, _ArrayLike_Callable, None] = None,
        integration_method=integ.quad,
    ):
        """Spectral Distortion."""
        super().__init__(cosmo)
        self.class_cosmo = class_cosmo  # TODO maybe move to superclass

        if AkFunc is None:
            self.AkFunc: _ArrayLike_Callable = default_Ak.get()
        elif isinstance(AkFunc, str):
            with default_Ak.set(AkFunc):
                self.AkFunc: _ArrayLike_Callable = default_Ak.get()

        self.emission_residual = None
        self.angular_residual = None

        # zeta array
        # TODO methods to set zeta array?
        thermo = class_cosmo.get_thermodynamics()
        self._zeta_arr = self.zeta(thermo["z"])

    # /def

    # ------------------------------

    @u.quantity_input(freq="frequency", kvec=1 / u.Mpc)
    def prefactor(
        self, freq: u.Quantity, kvec: u.Quantity,
    ):
        r"""Combined Prefactor.

        .. |quantity| replace:: `~astropy.units.Quantity`

        Parameters
        ----------
        kvec : |quantity|
            Units of inverse Megaparsec.
        PgamBarCL0 : float
            PgamBarCL at zeta0
        AkFunc : Callable or str or None
            Function to get amplitude :math:`A(\vec{k})`
            If str or None, uses `~default_Ak`
        Tcmb0 : |quantity|
            Temperature in Kelvin.

        """
        reduced_energy = freq * const.h / (const.k_B * self.Tcmb0) << u.one
        prefactor = (
            reduced_energy
            / np.expm1(-reduced_energy)
            * self.lambda0 ** 2
            * self.AkFunc(kvec)
            / (16 * np.pi * self.PgamBarCL0)
        )

        return prefactor

    # /def


# /class


#####################################################################


class SpectralDistortion(SpectralDistortionBase):
    """Spectral Distortion.

    Parameters
    ----------
    cosmo : `~astropy.cosmology.Cosmology`
    GgamBarCL : Callable
    PgamBarCL : Callable
    AkFunc: Callable or str or None, optional

    """

    def __init__(
        self,
        cosmo,
        class_cosmo,
        *,
        AkFunc: T.Union[str, _ArrayLike_Callable, None] = None,
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
        self.PgamBarCL: _IUSType = IUS(self._zeta_arr, thermo["exp(-kappa)"])
        self.GgamBarCL: _IUSType = IUS(self._zeta_arr, thermo["g [Mpc^-1]"])

        self.PgamBarCL0: float = self.PgamBarCL(self.zeta0)

    # /def

    # ------------------------------

    def _emission_integrand(
        self,
        zetaE: float,
        zetaS: float,
        # other variables
        k: np.ndarray,
        rEShat: np.ndarray,
    ):
        r"""Emission Integrand.

        :math:`g(\zeta_E) * \exp{(i \vec{k}\cdot \vec{r}_{SE}) / \sqrt((1+\zeta_E) {\zeta_E}^3)}`

        .. |ndarray| replace:: `~numpy.ndarray`
        .. |quantity| replace:: `~astropy.units.Quantity`

        Parameters
        ----------
        zetaE, zetaS : float
            zeta at emission and scatter, respectively

        Returns
        -------
        float
            Integrand, shown above.

        Other Parameters
        ----------------
        k : |quantity|
            Vector k, inverse Mpc
        rSEhat : |ndarray|
            Direction from zetaE to zetaS

        Notes
        -----
        .. todo::

            Double check I did rSE, rES substitution correctly

            Double check there is never an imaginary part

            convert rMag to produce

        """
        emission_integrand = self.GgamBarCL(zetaE) * np.exp(
            1j * self._rMag_Mpc(zetaE, zetaS) * k.dot(rEShat)
        ) / np.sqrt((1.0 + zetaE) * zetaE ** 3)

        return np.real(emission_integrand)

    # /def

    def emission_integral(
        self,
        thetaES: float,
        phiES: float,
        zetaS: float,
        *,
        k: np.ndarray,
        zetaMax: float = np.inf,
        **integration_kwargs,
    ):
        """Emission Integral.

        .. |quantity| replace:: `~astropy.units.Quantity`

        Parameters
        ----------
        thetaES, phiES : float
            In radians.
        zetaS : float
            zeta at scatter

        Returns
        -------
        float

        Other Parameters
        ----------------
        zetaMax : float
            Maximum zeta
        k : |quantity|
            Vector k, inverse Mpc
        integration_kwargs
            Keyword arguments into integration.
            Need to set ``epsabs``. 1e-9 is a good value.

        """
        # calculate rEShat for use in emission_integrand
        rEShat = self.rEShat(thetaES, phiES)

        # integrate emission integrand
        integral, residual = integ.quad(
            self._emission_integrand,
            zetaS,
            zetaMax,
            args=(zetaS, k, rEShat),
            **integration_kwargs,
        )
        return integral, residual

    # ------------------------------

    def _angular_integrand(
        self,
        thetaES: float,
        phiES: float,
        zetaS: float,
        # other arguments
        k: np.ndarray,
        zetaMax: float = np.inf,
        **integration_kwargs,
    ):
        """Angular Integrand.

        .. |quantity| replace:: `~astropy.units.Quantity`

        Parameters
        ----------
        thetaES, phiES : float
            In radians.
        zetaS : float
            zeta at scatter

        Returns
        -------
        float

        Other Parameters
        ----------------
        zetaMax : float
            Maximum zeta
        k : |quantity|
            Vector k, inverse Mpc
        integration_kwargs
            Keyword arguments into integration.
            ``epsabs`` is set to 1e-9 if not provided.

        """
        integration_kwargs["epsabs"] = integration_kwargs.get("epsabs", 1e-9)
        integral, _ = self.emission_integral(
            thetaES, phiES, zetaS, k=k, zetaMax=zetaMax, **integration_kwargs
        )

        return (
            (1.0 + np.cos(thetaES) ** 2)
            * integral
            * np.sin(thetaES)  # from d\Omega
        )

    # /def

    def angular_integral(
        self,
        zetaS: float,
        *,
        k: np.ndarray,
        zetaMax: float = np.inf,
        **integration_kwargs,
    ):
        """Angular Integrand.

        .. |quantity| replace:: `~astropy.units.Quantity`

        Parameters
        ----------
        thetaES, phiES : float
            In radians.
        zetaS : float
            zeta at scatter

        Returns
        -------
        float

        Other Parameters
        ----------------
        zetaMax : float
            Maximum zeta
        k : |quantity|
            Vector k, inverse Mpc
        integration_kwargs
            Keyword arguments into integration.
            ``epsabs`` is set to 1e-9 if not provided.

        """
        integration_kwargs["epsabs"] = integration_kwargs.get("epsabs", 1e-9)

        integral, residual = integ.dblquad(
            self._angular_integrand,
            0,
            2 * np.pi,  # Phi bounds
            lambda x: 0,
            lambda x: np.pi,  # theta bounds
            args=(zetaS, k),
            **integration_kwargs,
        )

        return integral, residual

    # /def

    # ------------------------------

    def _scatter_integrand(
        self,
        zetaS: float,
        # other variables
        k: u.Quantity,
        zeta0: float,
        zetaMax: float = np.inf,
        **integration_kwargs,
    ):
        """Scatter Integrand.

        .. |quantity| replace:: `~astropy.units.Quantity`

        Parameters
        ----------
        zetaS : float

        Returns
        -------
        float

        Other Parameters
        ----------------
        k : |quantity|
        zeta0 : float

        Notes
        -----
        .. todo::

            Check integration bounds

            Make it work on floats, not Quantities

        """
        kdotzhat = k[2]
        rS = self.rMag(zetaS, zeta0)
        g = self.GgamBarCL(zetaS) / self.PgamBarCL(zetaS)

        integral, _ = self.angular_integral(
            zetaS, k=k, zetaMax=zetaMax, **integration_kwargs
        )

        return (
            g
            * np.exp(1j * rS * kdotzhat)
            / np.sqrt((1.0 + zetaS) * zetaS ** 3)
            * integral
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


##############################################################################
# END
