# -*- coding: utf-8 -*-

"""Convert between different distance measures.

.. todo::

    speed up calculations by hard coding.

"""

__all__ = [
    "z_matter_radiation_equality",
    "z_of",
    "a_of",
    "alpha_observer",
    "alpha_of",
    "rho_of",
]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T

# THIRD PARTY
import numpy as np
import astropy.units as u
from astropy.cosmology import default_cosmology
from astropy.cosmology.core import Cosmology

# PROJECT-SPECIFIC
from cosmic_conchometer.typing import ArrayLike, TArrayLike


##############################################################################
# PARAMETERS

# alpha
alpha0: u.Quantity
alpha_0: u.Quantity
alpha0 = alpha_0 = 0 << u.one

alphaeq: u.Quantity
alpha_eq: u.Quantity
alphaeq = alpha_eq = 0 << u.one

# rho
rho0: u.Quantity

# types
TZ = T.TypeVar("TZ", float, u.Quantity)  # u.Quantity[dimensionless]

##############################################################################
# CODE
##############################################################################


def z_matter_radiation_equality(
    cosmo: Cosmology,
    zmin: TZ = 1e3,
    zmax: TZ = 1e4,
    *,
    full_output: bool = False,
    **rootkw: T.Any,
) -> T.Tuple[u.Quantity, T.Tuple[T.Any, ...]]:
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
        diff: TZ = cosmo.Om(z) - cosmo.Ogamma(z)
        return diff

    z_eq: u.Quantity
    rest: T.Tuple[T.Any, ...]
    zeq, *rest = brentq(f, zmin, zmax, full_output=True, **rootkw)
    z_eq = zeq << u.one

    return z_eq if not full_output else (z_eq, rest)


# /def


class z_of:
    """Calculate redshift from...  (time moves forward)."""

    @staticmethod
    def a(a: TArrayLike) -> TArrayLike:
        """Redshift from the scale factor."""
        z: TArrayLike = (1.0 / a) - 1.0
        return z

    @staticmethod
    def alpha(alpha: TArrayLike, **eq: TZ) -> TArrayLike:
        """Redshift from alpha.

        Parameters
        ----------
        alpha: array-like
        **eq : float or scalar `~astropy.units.Quantity`
            works with keys "aeq" or "zeq"

        Returns
        -------
        z : array-like or quantity-like ['dimensionless']

        Raises
        ------
        KeyError
            If 'eq' does not have key "aeq" nor "zeq".

        """
        aeq: TZ = eq.get("aeq", 1.0 / (eq.get("zeq") + 1.0))
        # hardcoded a_of.z(zeq)

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
            works with keys "aeq" or "zeq"
        
        """
        aeq: TZ = eq.get("aeq", 1.0 / (eq.get("zeq") + 1.0))

        alpha: TArrayLike = 2.0 * rho ** 2 - 1.0
        a: TArrayLike = aeq * alpha  # hardcoded a_of.alpha(alpha, aeq=aeq)
        z: TArrayLike = (1.0 / a) - 1.0  # hardcoded z_of.a(a)
        return z

    # -----------------------------------
    # specific times

    @property
    def naught(self) -> u.Quantity:
        """Redshift at the beginning of time."""
        z: u.Quantity = np.inf << u.one
        return z

    @staticmethod
    def matter_radiation_equality(
        cosmo: Cosmology, zmin: TZ = 1e3, zmax: TZ = 1e4, **rootkw: T.Any
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
        rootkw["full_output"] = False  # ensure output
        z_eq: u.Quantity = z_matter_radiation_equality(
            cosmo=cosmo, zmin=zmin, zmax=zmax, **rootkw
        )
        return z_eq

    @property
    def observer(self) -> u.Quantity:  # u.Quantity['dimensionless']
        """Redshift at our location."""
        z = 0.0 << u.one
        return z


# /class


##############################################################################
# Scale factor


class a_of:
    """Scale factor from...  (time moves forward)."""

    @staticmethod
    def z(z: TArrayLike) -> TArrayLike:
        """Scale factor from the redshift."""
        return 1.0 / (z + 1.0)

    @staticmethod
    def alpha(alpha: TArrayLike, aeq: TZ) -> TArrayLike:
        """Scale factor from alpha."""
        return aeq * alpha

    @staticmethod
    def rho(rho: TArrayLike, **eq: TZ) -> TArrayLike:
        """Scale factor from rho.

        Parameters
        ----------
        rho: array-like
        **eq : float or scalar `~astropy.units.Quantity`
            works with keys "aeq" or "zeq"
        
        """
        aeq: TZ = eq.get("aeq", 1.0 / (eq.get("zeq") + 1.0))

        alpha: TArrayLike = 2.0 * rho ** 2 - 1.0
        a: TArrayLike = aeq * alpha  # hardcoded a_of.alpha(alpha, aeq=aeq)
        return a

    # -----------------------------------
    # specific times

    @property
    def naught(self) -> u.Quantity:
        """Scale factor at the beginning of time."""
        return 0.0 << u.one

    @staticmethod
    def matter_radiation_equality(
        cosmo: Cosmology, amin: float = 0.0, amax: float = 0.1, **rootkw: T.Any
    ) -> u.Quantity:
        """Scale factor at matter-radiation equality."""
        zeq: u.Quantity = z_matter_radiation_equality(
            cosmo=cosmo, zmin=z_of.a(amin), zmax=z_of.a(amax), **rootkw
        )
        aeq: u.Quantity = 1.0 / (zeq + 1.0)
        return aeq

    @property
    def observer(self) -> u.Quantity:
        """Scale factor at our location."""
        return 1.0 << u.one


# class


##############################################################################
# alpha


def alpha_observer(cosmo: Cosmology, **zeq_kw) -> u.Quantity:
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
    zeq = z_matter_radiation_equality(cosmo, **zeq_kw)

    return zeq + 1.0


# /def


class alpha_of:
    """Calculate alpha from...  (time moves forward)."""

    @staticmethod
    def a(a: TArrayLike, **eq: TZ) -> TArrayLike:
        """alpha from the scale factor."""
        aeq: TZ = eq.get("aeq", 1.0 / (eq.get("zeq") + 1.0))
        alpha: TArrayLike = a / aeq
        return alpha

    @staticmethod
    def z(z: TArrayLike, **eq: TZ) -> TArrayLike:
        """alpha from the redshift."""
        aeq: TZ = eq.get("aeq", 1.0 / (eq.get("zeq") + 1.0))
        a: TArrayLike = 1.0 / (z + 1.0)
        alpha: TArrayLike = a / aeq
        return alpha

    @staticmethod
    def rho(rho: TArrayLike, **eq: TZ) -> TArrayLike:
        """alpha from rho.

        Parameters
        ----------
        rho: array-like
        **eq : float or scalar `~astropy.units.Quantity`
            works with keys "aeq" or "zeq"
        
        """
        aeq: TZ = eq.get("aeq", 1.0 / (eq.get("zeq") + 1.0))

        alpha: TArrayLike = 2.0 * rho ** 2 - 1.0
        return alpha

    # -----------------------------------
    # specific times

    @property
    def naught(self) -> u.Quantity:
        """alpha at the beginning of time."""
        return 0.0 << u.one

    @property
    def matter_radiation_equality(self) -> u.Quantity:
        """alpha at matter-radiation equality."""
        return 1.0 << u.one

    @staticmethod
    def observer(cosmo: Cosmology, **zeq_kw) -> u.Quantity:
        """alpha at our location."""
        return alpha_observer(cosmo, **zeq_kw)


# /class


##############################################################################
# rho


class rho_of:
    """Calculate rho from...  (time moves forward)."""

    @staticmethod
    def z(z: TArrayLike, zeq: float) -> TArrayLike:
        """rho from the redshift."""
        return np.sqrt((1.0 + alpha_of.z(z, zeq=zeq)) / 2.0)

    @staticmethod
    def a(a: TArrayLike, aeq: float) -> TArrayLike:
        """rho from the scale factor."""
        return np.sqrt((1.0 + a / aeq) / 2.0)

    @staticmethod
    def alpha(alpha: TArrayLike) -> TArrayLike:
        """rho from alpha."""
        return np.sqrt((1.0 + alpha) / 2.0)

    # -----------------------------------
    # specific times

    @property
    def naught(self) -> u.Quantity:
        """rho at the beginning of time."""
        return 1.0 / np.sqrt(2.0) << u.one

    @property
    def matter_radiation_equality(self) -> u.Quantity:
        """rho at matter-radiation equality."""
        return 1.0 << u.one

    @staticmethod
    def observer(cosmo: Cosmology, **zeq_kw) -> u.Quantity:
        """rho at our location."""
        alpha: u.Quantity = alpha_observer(cosmo, **zeq_kw)
        rho: u.Quantity = np.sqrt((1.0 + alpha) / 2.0)
        return rho


# /class

##############################################################################
# r

# # TODO! less sure of this one. is it comonent dependent?
# class r_of:


##############################################################################
# END
