# -*- coding: utf-8 -*-

"""Core functions safe to be used in lower modules."""

__all__ = [
    # functions
    "_blackbody",
    "blackbody",
    "default_Ak",
]


##############################################################################
# IMPORTS

# BUILT-IN

import typing as T


# THIRD PARTY

import numpy as np

import astropy.constants as const
import astropy.units as u
from astropy.cosmology import default_cosmology
from astropy.utils.state import ScienceState


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
def blackbody(freq: u.Quantity, temp: u.Quantity) -> _bb_unit:
    """Blackbody.

    .. |quantity| replace:: `~astropy.units.Quantity`

    Parameters
    ----------
    freq : |quantity|
    temp : |quantity|

    Returns
    -------
    |quantity|
        The blackbody in units of ``erg / (cm ** 2 * s * Hz * sr)``

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
    return 1.0


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
            cls._value: T.Callable = value
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


##############################################################################
# END
