# -*- coding: utf-8 -*-

"""Convert between different distance measures.

.. todo::

    speed up calculations by hard coding.

"""

##############################################################################
# IMPORTS

from __future__ import annotations

# BUILT-IN
import typing as T

# THIRD PARTY
import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.cosmology import default_cosmology
from astropy.cosmology.core import Cosmology

# PROJECT-SPECIFIC
from cosmic_conchometer.typing import TArrayLike

__all__ = [
    "lambda_naught",
    "z_matter_radiation_equality",
    "z_of",
    "a_of",
    "alpha_observer",
    "alpha_of",
    "rho_of",
]

##############################################################################
# PARAMETERS

# alpha
alpha0: u.Quantity = 0
alphaeq: u.Quantity = 1

# rho
rho0: u.Quantity = 1 / np.sqrt(2)
rhoeq: u.Quantity = 1

# static types
TZ = T.TypeVar("TZ", float, u.Quantity)  # u.Quantity[dimensionless]

##############################################################################
# CODE
##############################################################################


# @u.quantity_input(returns=u.Mpc)
def lambda_naught(cosmo: Cosmology, **zeq_kw: T.Any) -> u.Quantity:
    """Compute lambda0.

    Parameters
    ----------
    cosmo : `~astropy.cosmology.Cosmology`
    **zeq_kw
        kwargs into scipy minimizer. See `~scipy.optimize.brentq` for details.

    """
    alphaobs: u.Quantity = alpha_observer(cosmo, **zeq_kw)
    lambda0: u.Quantity
    lambda0 = (const.c / cosmo.H0) * np.sqrt(8 / (cosmo.Om0 * alphaobs))
    return lambda0 << u.Mpc


# /def


def _aeq(kw: T.Dict[str, T.Any]) -> TZ:
    if "aeq" in kw:
        aeq = kw["aeq"]
    elif "zeq" in kw:
        aeq = _A_Of.z(kw["zeq"])
    else:
        aeq = _A_Of.matter_radiation_equality(kw["cosmo"])
    return aeq


# /def

##############################################################################


def z_matter_radiation_equality(
    cosmo: Cosmology,
    zmin: TZ = 1e3,
    zmax: TZ = 1e4,
    *,
    full_output: bool = False,
    **rootkw: T.Any,
) -> T.Union[u.Quantity, tuple[u.Quantity, tuple[T.Any, ...]]]:
    """Calculate matter-radiation equality redshift for a given cosmology.

    This works for a cosmology with any number of components.
    If the matter-radiation equality redshift can be calculated analytically,
    that should be used instead of this general function.

    Parameters
    ----------
    cosmo : `~astropy.cosmology.Cosmology`
        Must have methods ``Om(z)`` and ``Ogamma(z)``.
    zmin, zmax : float or scalar `~astropy.units.Quantity` (optional)
        The min/max redshift in which to search for matter-radiation equality.
    full_output : bool (optional, keyword-only)
        Whether to return the full output of the scipy rootfinder,
        or just matter-radiation equality.
    **rootkw
        kwargs into scipy minimizer. See `~scipy.optimize.brentq` for details.

    Returns
    -------
    z_eq : scalar `~astropy.units.Quantity` [dimensionless]
        The redshift at matter-radiation equality.
    tuple
        Full results of root finder if `full_output` is True

    """
    # THIRD PARTY
    from scipy.optimize import brentq

    # residual function
    def f(z: TZ) -> TZ:
        diff: TZ = cosmo.Om(z) - (cosmo.Ogamma(z) + cosmo.Onu(z))
        return diff

    rest: tuple[T.Any, ...]
    zeq, *rest = brentq(f, zmin, zmax, full_output=True, **rootkw)
    z_eq: u.Quantity = zeq

    return z_eq if not full_output else (z_eq, rest)


# /def


class _Z_Of:
    """Calculate redshift from...  (time moves forward)."""

    @staticmethod
    def a(a: TArrayLike) -> TArrayLike:
        """Redshift from the scale factor."""
        z: TArrayLike = np.divide(1.0, a) - 1.0
        return z

    @staticmethod
    def alpha(alpha: TArrayLike, **eq: TZ) -> TArrayLike:
        """Redshift from alpha.

        Parameters
        ----------
        alpha: array-like
        **eq : float or scalar `~astropy.units.Quantity`
            works with keys "aeq" or "zeq" or "cosmo"

        Returns
        -------
        z : array-like or quantity-like ['dimensionless']

        """
        aeq: TZ = _aeq(eq)

        a: TArrayLike = aeq * alpha  # hardcoded a_of.alpha(alpha, aeq=aeq)
        z: TArrayLike = (1.0 / a) - 1.0  # hardcoded z_of.a(a)
        return z

    @staticmethod
    def rho(rho: TArrayLike, **eq: TZ) -> TArrayLike:
        """z from rho.

        Parameters
        ----------
        rho: array-like
        **eq : float or scalar `~astropy.units.Quantity`
            works with keys "aeq" or "zeq" or "cosmo"

        """
        aeq: TZ = _aeq(eq)

        alpha: TArrayLike = 2.0 * rho ** 2 - 1.0
        a: TArrayLike = aeq * alpha  # hardcoded a_of.alpha(alpha, aeq=aeq)
        z: TArrayLike = (1.0 / a) - 1.0  # hardcoded z_of.a(a)
        return z

    # -----------------------------------
    # specific times

    @property
    def naught(self) -> u.Quantity:
        """Redshift at the beginning of time."""
        z: u.Quantity = np.inf
        return z

    @staticmethod
    def matter_radiation_equality(
        cosmo: T.Optional[Cosmology] = None,
        zmin: TZ = 1e3,
        zmax: TZ = 1e4,
        **rootkw: T.Any,
    ) -> u.Quantity:
        """Redshift at matter-radiation equality.

        Parameters
        ----------
        cosmo : `~astropy.cosmology.Cosmology`
            Must have methods ``Om(z)`` and ``Ogamma(z)``.
        zmin, zmax : float or scalar `~astropy.units.Quantity` (optional)
            The min/max redshift in which to search.
        **rootkw
            kwargs into minimizer. See `~scipy.optimize.brentq` for details.

        Returns
        -------
        z_eq : `~astropy.units.Quantity`
            The redshift at matter-radiation equality.

        """
        if cosmo is None:
            cosmo = default_cosmology.get()

        rootkw["full_output"] = False  # ensure output
        z_eq: u.Quantity = z_matter_radiation_equality(
            cosmo=cosmo, zmin=zmin, zmax=zmax, **rootkw
        )
        return z_eq

    @property
    def observer(self) -> u.Quantity:  # u.Quantity['dimensionless']
        """Redshift at our location."""
        z = 0.0
        return z


z_of = _Z_Of()
# /class


##############################################################################
# Scale factor


class _A_Of:
    """Scale factor from...  (time moves forward)."""

    @staticmethod
    def z(z: TArrayLike) -> TArrayLike:
        """Scale factor from the redshift."""
        return 1.0 / (z + 1.0)

    @staticmethod
    def alpha(alpha: TArrayLike, aeq: TZ) -> TArrayLike:
        """Scale factor from alpha."""
        a: TArrayLike = aeq * alpha
        return a

    @staticmethod
    def rho(rho: TArrayLike, **eq: TZ) -> TArrayLike:
        """Scale factor from rho.

        Parameters
        ----------
        rho: array-like
        **eq : float or scalar `~astropy.units.Quantity`
            works with keys "aeq" or "zeq"

        """
        aeq: TZ = _aeq(eq)

        alpha: TArrayLike = 2.0 * rho ** 2 - 1.0
        a: TArrayLike = aeq * alpha  # hardcoded a_of.alpha(alpha, aeq=aeq)
        return a

    # -----------------------------------
    # specific times

    @property
    def naught(self) -> u.Quantity:
        """Scale factor at the beginning of time."""
        return 0.0

    @staticmethod
    def matter_radiation_equality(
        cosmo: T.Optional[Cosmology] = None,
        amin: float = 0.0,
        amax: float = 0.1,
        **rootkw: T.Any,
    ) -> u.Quantity:
        """Scale factor at matter-radiation equality."""
        if cosmo is None:
            cosmo = default_cosmology.get()

        zeq: u.Quantity = z_matter_radiation_equality(
            cosmo=cosmo, zmin=z_of.a(amin), zmax=z_of.a(amax), **rootkw
        )
        aeq: u.Quantity = 1.0 / (zeq + 1.0)
        return aeq

    @property
    def observer(self) -> u.Quantity:
        """Scale factor at our location."""
        return 1.0


a_of = _A_Of()
# class


##############################################################################
# alpha


def alpha_observer(cosmo: Cosmology, **zeq_kw: T.Any) -> u.Quantity:
    """alpha at the location of the observer.

    Parameters
    ----------
    cosmo : `~astropy.cosmology.Cosmology`
        Must have methods ``Om(z)`` and ``Ogamma(z)``.
    **zeq_kw
        kwargs into scipy minimizer. See `~scipy.optimize.brentq` for details.

    Returns
    -------
    `~astropy.units.Quantity`

    """
    zeq_kw["full_output"] = False
    zeq: u.Quantity = z_matter_radiation_equality(cosmo, **zeq_kw)

    return zeq + 1.0


# /def


class _Alpha_Of:
    """Calculate alpha from...  (time moves forward)."""

    @staticmethod
    def a(a: TArrayLike, **eq: TZ) -> TArrayLike:
        """alpha from the scale factor."""
        aeq: TZ = _aeq(eq)
        alpha: TArrayLike = a / aeq
        return alpha

    @staticmethod
    def z(z: TArrayLike, **eq: TZ) -> TArrayLike:
        """alpha from the redshift."""
        aeq: TZ = _aeq(eq)
        a: TArrayLike = 1.0 / (z + 1.0)
        alpha: TArrayLike = a / aeq
        return alpha

    @staticmethod
    def rho(rho: TArrayLike) -> TArrayLike:
        """alpha from rho.

        Parameters
        ----------
        rho: array-like or quantity-like ['dimensionless']

        """
        alpha: TArrayLike = 2.0 * rho ** 2 - 1.0
        return alpha

    # -----------------------------------
    # specific times

    @property
    def naught(self) -> u.Quantity:
        """alpha at the beginning of time."""
        return alpha0

    @property
    def matter_radiation_equality(self) -> u.Quantity:
        """alpha at matter-radiation equality."""
        return alphaeq

    @staticmethod
    def observer(cosmo: Cosmology, **zeq_kw: T.Any) -> u.Quantity:
        """alpha at our location."""
        return alpha_observer(cosmo, **zeq_kw)


alpha_of = _Alpha_Of()
# /class


##############################################################################
# rho


class _Rho_Of:
    """Calculate rho from...  (time moves forward)."""

    @staticmethod
    def z(z: TArrayLike, **eq: TZ) -> TArrayLike:
        """rho from the redshift."""
        rho: TArrayLike = np.sqrt((1.0 + alpha_of.z(z, **eq)) / 2.0)
        return rho

    @staticmethod
    def a(a: TArrayLike, **eq: float) -> TArrayLike:
        """rho from the scale factor."""
        aeq: TZ = _aeq(eq)
        rho: TArrayLike = np.sqrt((1.0 + a / aeq) / 2.0)
        return rho

    @staticmethod
    def alpha(alpha: TArrayLike) -> TArrayLike:
        """rho from alpha."""
        rho: TArrayLike = np.sqrt((1.0 + alpha) / 2.0)
        return rho

    # -----------------------------------
    # specific times

    @property
    def naught(self) -> u.Quantity:
        """rho at the beginning of time."""
        return rho0

    @property
    def matter_radiation_equality(self) -> u.Quantity:
        """rho at matter-radiation equality."""
        return rhoeq

    @staticmethod
    def observer(cosmo: Cosmology, **zeq_kw: T.Any) -> u.Quantity:
        """rho at our location."""
        alpha: u.Quantity = alpha_observer(cosmo, **zeq_kw)
        rho: u.Quantity = np.sqrt((1.0 + alpha) / 2.0)
        return rho


rho_of = _Rho_Of()
# /class


##############################################################################
# END
