# -*- coding: utf-8 -*-

"""Core functions safe to be used in lower modules."""

__all__ = [
    # functions
    "_blackbody",
    "blackbody",
    "default_Ak",
    "CosmologyDependent",
]


##############################################################################
# IMPORTS

import typing as T

import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.cosmology import default_cosmology
from astropy.cosmology.core import Cosmology
from astropy.utils.state import ScienceState

from .utils import (
    z_matter_radiation_equality,
    zeta_of_z as _zeta_of_z,
    z_of_zeta as _z_of_zeta,
)


##############################################################################
# PARAMETERS

with default_cosmology.set("Planck18_arXiv_v2"):
    default_cosmo = default_cosmology.get()


# Blackbody
_bb_unit = u.erg / (u.cm ** 2 * u.s * u.Hz * u.sr)  # default unit

_GHz_h_kB_in_K = (1 * u.GHz * const.h / const.k_B).to_value(u.K)
"""h / k_B times GHz."""
_GHz3_h_c2_in_erg_Hzssrcm2 = (
    1 * u.GHz ** 3 * const.h / const.c ** 2 / u.sr
).to_value(_bb_unit)
"""h / c**2 times GHz^3 in erg / Hz s sr cm^2."""

QuantityType = T.TypeVar("QuantityType", u.Quantity, u.SpecificTypeQuantity)


##############################################################################
# CODE
##############################################################################
# Blackbody


def _blackbody(
    freq: T.Union[float, np.ndarray], temp: T.Union[float, np.ndarray]
):
    """Blackbody, without units.

    Parameters
    ----------
    freq : array-like
        Frequency in GHz.
    temp : array-like
        Temperature in K.

    Returns
    -------
    array-like
        The blackbody in units of ``erg / (cm ** 2 * s * Hz * sr)``

    """
    log_boltz = _GHz_h_kB_in_K * freq / temp
    boltzm1 = np.expm1(log_boltz)

    bb_nu = 2.0 * _GHz3_h_c2_in_erg_Hzssrcm2 * freq ** 3 / boltzm1

    return bb_nu


# /def


# -------------------------------------------------------------------


@u.quantity_input(freq="frequency", temp=u.K)
def blackbody(freq: QuantityType, temp: QuantityType) -> _bb_unit:
    """Blackbody.

    Parameters
    ----------
    freq : |quantity|
    temp : |quantity|

    Returns
    -------
    |quantity|
        The blackbody in units of ``erg / (cm ** 2 * s * Hz * sr)``

    ..
      RST SUBSTITUTIONS

    .. |quantity| replace:: `~astropy.units.Quantity`

    """
    return _blackbody(freq.to_value(u.GHz), temp.to_value(u.K)) * _bb_unit


# /def


#####################################################################


def _Ak_unity(k: np.ndarray) -> float:
    """:math:`A(k) = 1.0`.

    Parameters
    ----------
    k : `~numpy.ndarray`
        k vector.

    Returns
    -------
    float
        1.0

    """
    return 1.0 + 0.0j


# /def


# -------------------------------------------------------------------


class default_Ak(ScienceState):
    """Default Ak.

    Methods
    -------
    get_from_str
        Get Ak function from registry by name
    validate
        Validate value as Ak function
    register
        register Ak function

    """

    _default_value = "unity"
    """Default value for A(k)"""
    _value = None
    """Current value of A(k)"""

    _registry = {"unity": _Ak_unity}
    """Registry of A(k) functions."""

    @classmethod
    def get_from_str(cls, name: str) -> T.Callable:
        """Get Ak function from string.

        Parameters
        ----------
        name : str
            Name of Ak function in registry

        Returns
        -------
        Callable
            Ak function

        """
        return cls._registry[name]  # todo copy function?

    # /def

    @classmethod
    def validate(cls, value: T.Union[None, str, T.Callable]) -> T.Callable:
        """Validate Ak function.

        Parameters
        ----------
        name : str
            Name of Ak function in registry

        Returns
        -------
        Callable
            Ak function

        """
        if value is None:
            value: str = cls._default_value

        if isinstance(value, str):
            value: T.Callable = cls.get_from_str(value)
        elif callable(value):
            cls._state["value"]: T.Callable = value
        else:
            raise TypeError

        return value

    # /def

    @classmethod
    def register(cls, name: str, AkFunc: T.Callable, overwrite: bool = False):
        """Register Ak function.

        Parameters
        ----------
        name : str
            Name of Ak function for registry
        AkFunc : Callable
            Ak func
        overwrite : bool
            Whether to overwrite the Ak function, if already in the registry.

        """
        if name in cls._registry and not overwrite:
            raise ValueError(f"cannot overwrite {name}")
        cls._registry[name] = AkFunc

    # /def


# /class


#####################################################################


class CosmologyDependent:
    """Class for coordinating cosmology-dependent calculations.

    Parameters
    ----------
    cosmo : ``astropy.cosmology.Cosmology`` instance

    ..
      RST SUBSTITUTIONS

    .. |NDarray| replace:: `~numpy.ndarray`
    .. |Quantity| replace:: `~astropy.units.Quantity`

    """

    def __init__(self, cosmo: Cosmology):
        self._cosmo = cosmo

        self.Tcmb0: QuantityType = cosmo.Tcmb0
        self.zeq: float = z_matter_radiation_equality(cosmo)
        self.zeta0: float = self.zeta(0)

        # TODO? make this a stand-alone function
        self.lambda0 = (
            (const.c / cosmo.H0) * np.sqrt(self.zeta0 / cosmo.Om0)
        ) << u.Mpc
        self._lambda0_Mpc = self.lambda0.to_value(u.Mpc)

    # /def

    @property
    def cosmo(self) -> Cosmology:
        """Read-only cosmology."""
        return self._cosmo

    # /def

    # ------------------------------
    # Redshifts

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
        return _zeta_of_z(z, self.zeq)

    # /def

    def z_of_zeta(
        self, zeta: T.Union[float, np.ndarray]
    ) -> T.Union[float, np.ndarray]:
        """z(zeta).

        Parameters
        ----------
        zeta : float or array

        Returns
        -------
        z : float or array
            same as `zeta` type.

        """
        return _z_of_zeta(zeta, self.zeq)  # TODO z_of_zeta(function)

    # /def

    # ------------------------------
    # Distances

    def _rMag_Mpc(
        self, zeta1: T.Union[float, np.array], zeta2: T.Union[float, np.array]
    ) -> T.Union[float, np.array]:
        """Magnitude of r.

        .. |NDarray| replace:: `~numpy.ndarray`

        Parameters
        ----------
        zeta1, zeta2 : float or |NDarray|

        Returns
        -------
        float or |NDarray|
            float unless `zeta1` or `zeta2` is |NDarray|

        """
        return (
            2
            * self._lambda0_Mpc
            * (np.sqrt((zeta2 + 1.0) / zeta2) - np.sqrt((zeta1 + 1.0) / zeta1))
        )

    # /def

    def rMag(
        self, zeta1: T.Union[float, np.array], zeta2: T.Union[float, np.array]
    ) -> QuantityType:
        """Magnitude of r.

        Parameters
        ----------
        zeta1, zeta2 : float or |NDarray|

        Returns
        -------
        float or |NDarray|
            float unless `zeta1` or `zeta2` is |NDarray|

        """
        return self._rMag_Mpc(zeta1, zeta2) * u.Mpc

    # /def

    # @u.quantity_input(thetaES=u.deg, phiES=u.deg)  # not used
    @staticmethod
    def rEShat(
        thetaES: T.Union[float, np.ndarray, QuantityType],
        phiES: T.Union[float, np.ndarray, QuantityType],
    ) -> np.ndarray:
        """Direction from emission to scatter.

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

        ..
          RST SUBSTITUTIONS

        .. |ndarray| replace:: `~numpy.ndarray`
        .. |quantity| replace:: `~astropy.units.Quantity`

        """
        return np.array(
            [
                np.sin(thetaES) * np.cos(phiES),
                np.sin(thetaES) * np.sin(phiES),
                np.cos(thetaES),
            ]
        )

    # /def

    # ------------------------------
    # Static Methods

    @staticmethod
    def _blackbody(
        freq: T.Union[float, np.ndarray], temp: T.Union[float, np.ndarray]
    ) -> T.Union[float, np.ndarray]:
        """Blackbody spectrum, without units.

        Parameters
        ----------
        freq : array-like
            Frequency in GHz.
        temp : array-like
            Temperature in K.

        Returns
        -------
        array-like
            The blackbody in units of ``erg / (cm ** 2 * s * Hz * sr)``

        """
        return _blackbody(freq, temp)

    # /def

    @staticmethod
    @u.quantity_input(freq="frequency", temp=u.K)
    def blackbody(freq: QuantityType, temp: QuantityType) -> _bb_unit:
        """Blackbody spectrum.

        Parameters
        ----------
        freq : |quantity|
            Frequency in GHz.
        temp : |quantity|
            Temperature in K.

        Returns
        -------
        |quantity|
            The blackbody in units of ``erg / (cm ** 2 * s * Hz * sr)``

        """
        return blackbody(freq.to_value(u.GHz), temp.to_value(u.K)) * _bb_unit

    # /def

    @staticmethod
    def set_AkFunc(self, value: T.Union[None, str, T.Callable]):
        """Set the default function used in A(k).

        Can be used as a contextmanager.

        Parameters
        ----------
        value

        Returns
        -------
        ScienceStateContext
            Output of ``default_Ak.set``

        """
        return default_Ak.set(value)

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
