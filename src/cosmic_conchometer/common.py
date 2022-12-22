
"""Core functions safe to be used in lower modules."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import Callable, ClassVar, Optional, TypeVar, Union
from abc import ABCMeta
from collections.abc import Mapping


import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.cosmology import default_cosmology
from astropy.cosmology.core import Cosmology
from astropy.utils.metadata import MetaData
from astropy.utils.state import ScienceState

# PROJECT-SPECIFIC
from .config import conf
from .typing import ArrayLike
from .utils.distances import DistanceMeasureConverter

__all__ = [
    "default_cosmo",
    # functions
    "blackbody",
    "default_Ak",
    "CosmologyDependent",
]

##############################################################################
# PARAMETERS

# get the default cosmology from the configuration.
with default_cosmology.set(conf.default_cosmo):
    default_cosmo = default_cosmology.get()


# -----------
# Blackbody
_bb_unit: u.UnitBase = u.erg / (u.cm ** 2 * u.s * u.Hz * u.sr)  # default unit

# -----------

fAkType = Callable[[ArrayLike], complex]


##############################################################################
# CODE
##############################################################################
# Blackbody


# @u.quantity_input(freq="frequency", temp=u.K, returns=_bb_unit)
def blackbody(frequency: u.Quantity, temperature: u.Quantity) -> u.Quantity:
    r"""Planck blackbody function.

    Parameters
    ----------
    frequency : `~astropy.units.Quantity`
    temperature : `~astropy.units.Quantity`

    Returns
    -------
    `~astropy.units.Quantity`
        The blackbody in units of :math:`\frac{erg}{cm^2 sr}`

    """
    log_boltz = const.h * frequency / (const.k_B * temperature)
    boltzm1 = np.expm1(log_boltz)

    bb_nu = (2 * const.h / const.c ** 2 / u.sr) * frequency ** 3 / boltzm1
    return bb_nu << _bb_unit


#####################################################################


def _Ak_unity(k: ArrayLike) -> complex:
    """:math:`A(k) = 1.0 + 0j`.

    Parameters
    ----------
    k : array
        k vector.

    Returns
    -------
    Ak : complex
        (1 + 0j)

    """
    return 1.0 + 0.0j


# -------------------------------------------------------------------


class default_Ak(ScienceState):
    """Default Ak."""

    _default_value: str = "unity"
    """Default value for A(k)"""
    _value: Optional[fAkType] = None
    """Current value of A(k)"""

    _registry: ClassVar[dict[str, fAkType]] = {"unity": _Ak_unity}
    """Registry of A(k) functions."""

    @classmethod
    def get_from_str(cls, name: str) -> fAkType:
        """Get Ak function from string.

        Parameters
        ----------
        name : str
            Name of Ak function in registry.

        Returns
        -------
        callable[[array-like], complex]
            Ak function.

        """
        Akfunc = cls._registry[name]
        return Akfunc

    @classmethod
    def validate(cls, value: Union[None, str, Callable]) -> fAkType:
        """Validate Ak function.

        Parameters
        ----------
        name : str
            Name of Ak function in registry.

        Returns
        -------
        callable[[array-like], complex]
            Ak function.

        """
        if value is None:
            value = cls._default_value

        if isinstance(value, str):
            value = cls.get_from_str(value)
        elif callable(value):
            # cls._state["value"]: T.Callable = value
            cls._value = value
        else:
            raise TypeError

        return value

    @classmethod
    def register(
        cls,
        name: str,
        AkFunc: fAkType,
        overwrite: bool = False,
    ) -> None:
        """Register Ak function.

        Parameters
        ----------
        name : str
            Name of Ak function for registry
        AkFunc : callable
            Ak func
        overwrite : bool
            Whether to overwrite the Ak function, if already in the registry.

        """
        if name in cls._registry and not overwrite:
            raise ValueError(f"cannot overwrite {name}")
        cls._registry[name] = AkFunc


#####################################################################


class CosmologyDependent(metaclass=ABCMeta):
    """Class for coordinating cosmology-dependent calculations.

    Parameters
    ----------
    cosmo : `~astropy.cosmology.core.Cosmology` instance

    """

    meta = MetaData()

    def __init__(self, cosmo: Cosmology, *, meta: T.Optional[Mapping] = None):
        self._cosmo: Cosmology = cosmo  # the astropy cosmology
        self.meta.update(meta or {})

        self._distance_converter = DistanceMeasureConverter(cosmo)

    @property
    def cosmo(self) -> Cosmology:
        """Cosmology instance."""
        return self._cosmo

    @property
    def distance_converter(self) -> DistanceMeasureConverter:
        return self._distance_converter

    @property
    def lambda0(self) -> u.Quantity:
        return self._distance_converter.lambda0

    @property
    def z_eq(self) -> float:
        return self._distance_converter.z_matter_radiation_equality
