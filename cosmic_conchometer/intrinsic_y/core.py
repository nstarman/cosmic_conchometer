# -*- coding: utf-8 -*-

"""**DOCSTRING**.

Description.

"""

__all__ = [
    "CosmologyDependent",
]


##############################################################################
# IMPORTS

# BUILT-IN

import typing as T

# THIRD PARTY

import astropy.constants as const
import astropy.units as u
from astropy.cosmology import Cosmology

import numpy as np


# PROJECT-SPECIFIC

from cosmic_conchometer.utils import z_matter_radiation_equality, zeta_of_z


##############################################################################
# PARAMETERS


##############################################################################
# CODE
##############################################################################


class CosmologyDependent:
    """Class for coordinating cosmology-dependent calculations.

    Parameters
    ----------
    cosmo : `~astropy.cosmology.Cosmology` instance

    """

    def __init__(self, cosmo: Cosmology):
        self._cosmo = cosmo

        self.Tcmb0: u.Quantity = cosmo.Tcmb0
        self.zeq: float = z_matter_radiation_equality(cosmo)
        self.zeta0: float = self.zeta(0)

        self.lambda0 = (const.c / cosmo.H0) * np.sqrt(
            self.zeta0 / cosmo.Om0
        ) << u.Mpc
        self._lambda0_Mpc = self.lambda0.to_value(u.Mpc)

    # /def

    @property
    def cosmo(self) -> Cosmology:
        """Read-only cosmology."""
        return self._cosmo

    # /def

    # ------------------------------

    def zeta(
        self, z: T.Union[float, np.ndarray]
    ) -> T.Union[float, np.ndarray]:
        """Zeta(z).

        Parameters
        ----------
        z : float or array

        Returns
        -------
        zeta : float or array
            same as `z` type.

        """
        return zeta_of_z(z, self.zeq)

    # /def

    def z(
        self, zeta: T.Union[float, np.ndarray]
    ) -> T.Union[float, np.ndarray]:
        """Zeta(z).

        Parameters
        ----------
        zeta : float or array

        Returns
        -------
        z : float or array
            same as `zeta` type.

        """
        return zeta * (1.0 + self.zeq) - 1

    # /def

    # ------------------------------

    def _rMag_Mpc(
        self, zeta1: T.Union[float, np.array], zeta2: T.Union[float, np.array]
    ) -> T.Union[float, np.array]:
        """Magnitude of r.

        .. |ndarray| replace:: `~numpy.ndarray`

        Parameters
        ----------
        zeta1, zeta2 : float or |ndarray|

        Returns
        -------
        float or |ndarray|
            float unless `zeta1` or `zeta2` is |ndarray|

        """
        return (
            2
            * self._lambda0_Mpc
            * (np.sqrt((zeta2 + 1.0) / zeta2) - np.sqrt((zeta1 + 1.0) / zeta1))
        )

    # /def

    def rMag(
        self, zeta1: T.Union[float, np.array], zeta2: T.Union[float, np.array]
    ) -> u.Quantity:
        """Magnitude of r.

        .. |ndarray| replace:: `~numpy.ndarray`

        Parameters
        ----------
        zeta1, zeta2 : float or |ndarray|

        Returns
        -------
        float or |ndarray|
            float unless `zeta1` or `zeta2` is |ndarray|

        """
        return self._rMag_Mpc(zeta1, zeta2) * u.Mpc

    # /def

    # ------------------------------

    # @u.quantity_input(thetaES=u.deg, phiES=u.deg)  # not used
    @staticmethod
    def rEShat(
        thetaES: T.Union[float, np.ndarray, u.Quantity],
        phiES: T.Union[float, np.ndarray, u.Quantity],
    ):
        """Direction from emission to scatter.

        .. |quantity| replace:: `~astropy.units.Quantity`

        Parameters
        ----------
        thetaES, phiES : |quantity|
            In degrees

            .. todo::

                Explain theta and phi coordinates

        Returns
        -------
        |ndarray|
            (3,) array of direction from emission to scatter.
            Unit magnitude.

        """
        return np.array(
            [
                np.sin(thetaES) * np.cos(phiES),
                np.sin(thetaES) * np.sin(phiES),
                np.cos(thetaES),
            ]
        )

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
