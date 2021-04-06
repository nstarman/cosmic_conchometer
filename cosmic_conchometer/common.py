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

# BUILT-IN
import typing as T

# THIRD PARTY
import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.cosmology import default_cosmology
from astropy.cosmology.core import Cosmology
from astropy.utils.state import ScienceState

# PROJECT-SPECIFIC
from .config import conf
from .utils import z_matter_radiation_equality
from .utils import z_of_zeta as _z_of_zeta
from .utils import zeta_of_z as _zeta_of_z

##############################################################################
# PARAMETERS

# get the default cosmology from the configuration.
with default_cosmology.set(conf.default_cosmo):
    default_cosmo = default_cosmology.get()


# -----------
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
    freq: T.Union[float, np.ndarray],
    temp: T.Union[float, np.ndarray],
) -> T.Union[float, np.ndarray]:
    """Planck blackbody function, without units.

    Parameters
    ----------
    freq : array-like
        Frequency (value) in GHz.
    temp : array-like
        Temperature (value) in K.

    Returns
    -------
    array-like
        The blackbody (value) in units of :math:`\frac{erg}{cm^2 sr}``

    """
    log_boltz = _GHz_h_kB_in_K * freq / temp
    boltzm1 = np.expm1(log_boltz)

    bb_nu = 2.0 * _GHz3_h_c2_in_erg_Hzssrcm2 * freq ** 3 / boltzm1

    return bb_nu


# /def


# -------------------------------------------------------------------


@u.quantity_input(freq="frequency", temp=u.K, returns=_bb_unit)
def blackbody(freq: u.Quantity, temp: u.Quantity) -> u.Quantity:
    """Blackbody.

    Parameters
    ----------
    freq : `~astropy.units.Quantity`['frequency']
    temp : `~astropy.units.Quantity`['temperature']

    Returns
    -------
    `~astropy.units.Quanity`
        The blackbody in units of :math:`\frac{erg}{cm^2 sr}``

    """
    return _blackbody(freq.to_value(u.GHz), temp.to_value(u.K)) * _bb_unit


# /def


#####################################################################


def _Ak_unity(k: T.Union[float, np.ndarray]) -> complex:
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

    _default_value: str = "unity"
    """Default value for A(k)"""
    _value: T.Optional[T.Callable] = None
    """Current value of A(k)"""

    _registry: T.Dict[str, T.Callable] = {"unity": _Ak_unity}
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
        Akfunc = cls._registry[name]  # todo copy function?
        return Akfunc

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
        callable
            Ak function

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

    # /def

    @classmethod
    def register(
        cls,
        name: str,
        AkFunc: T.Callable,
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

    # /def


# /class


#####################################################################


class CosmologyDependent:
    """Class for coordinating cosmology-dependent calculations.

    Parameters
    ----------
    cosmo : `~astropy.cosmology.core.Cosmology` instance

    """

    def __init__(self, cosmo: Cosmology):
        self._cosmo = cosmo

        self.Tcmb0: u.Quantity = cosmo.Tcmb0
        self.zeq: float = z_matter_radiation_equality(cosmo)[0]
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
        self,
        z: T.Union[float, np.ndarray],
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
        self,
        zeta: T.Union[float, np.ndarray],
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
        self,
        zeta1: T.Union[float, np.array],
        zeta2: T.Union[float, np.array],
    ) -> T.Union[float, np.array]:
        """Magnitude of r.

        Parameters
        ----------
        zeta1, zeta2 : float or array

        Returns
        -------
        float or array
            float unless `zeta1` or `zeta2` is `~numpy.array`

        """
        return (
            2
            * self._lambda0_Mpc
            * (np.sqrt((zeta2 + 1.0) / zeta2) - np.sqrt((zeta1 + 1.0) / zeta1))
        )

    # /def

    def rMag(
        self,
        zeta1: T.Union[float, np.array],
        zeta2: T.Union[float, np.array],
    ) -> u.Quantity:
        """Magnitude of r.

        Parameters
        ----------
        zeta1, zeta2 : float or array

        Returns
        -------
        float or array
            float unless `zeta1` or `zeta2` is `~numpy.array`

        """
        return self._rMag_Mpc(zeta1, zeta2) * u.Mpc

    # /def

    # @u.quantity_input(thetaES=u.deg, phiES=u.deg)  # not used
    @staticmethod
    def rEShat(
        thetaES: T.Union[float, np.ndarray, u.Quantity],
        phiES: T.Union[float, np.ndarray, u.Quantity],
    ) -> np.ndarray:
        """Direction from emission to scatter.

        Parameters
        ----------
        thetaES, phiES : `~astropy.units.Quantity`
            In degrees

            .. todo::

                Explain theta and phi coordinates

        Returns
        -------
        array
            (3,) array of direction from emission to scatter.
            Unit magnitude.

        """
        return np.array(
            [
                np.sin(thetaES) * np.cos(phiES),
                np.sin(thetaES) * np.sin(phiES),
                np.cos(thetaES),
            ],
        )

    # /def

    # ------------------------------
    # Static Methods

    @staticmethod
    def _blackbody(
        freq: T.Union[float, np.ndarray],
        temp: T.Union[float, np.ndarray],
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
    @u.quantity_input(freq="frequency", temp=u.K, returns=_bb_unit)
    def blackbody(freq: u.Quantity, temp: u.Quantity) -> u.Quantity:
        """Blackbody spectrum.

        Parameters
        ----------
        freq : `~astropy.units.Quantity`
            Frequency in GHz.
        temp : `~astropy.units.Quantity`
            Temperature in K.

        Returns
        -------
        `~astropy.units.Quanity`
            The blackbody in units of ``erg / (cm ** 2 * s * Hz * sr)``

        """
        bb = blackbody(freq.to_value(u.GHz), temp.to_value(u.K)) * _bb_unit
        return bb

    # /def

    # @staticmethod
    def set_AkFunc(self, value: T.Union[None, str, T.Callable]) -> T.Callable:
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
        Akfunc: T.Callable = default_Ak.set(value)
        return Akfunc

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
