"""Convert between different distance measures."""

from __future__ import annotations

# STDLIB
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload
from weakref import ProxyType

# THIRD-PARTY
import astropy.constants as const
import astropy.cosmology.units as cu
import astropy.units as u
import numpy as np
import scipy.optimize as optim

if TYPE_CHECKING:
    # THIRD-PARTY
    from astropy.cosmology import FLRW
    from astropy.units import Quantity

__all__ = [
    "z_matter_radiation_equality",
    "DistanceMeasureConverter",
]

##############################################################################
# TYPING

# static types
TZ = TypeVar("TZ", float, "Quantity")  # u.Quantity[dimensionless]

##############################################################################
# CODE
##############################################################################


def _residual(z: TZ, cosmo: FLRW) -> TZ:
    return cosmo.Om(z) - (cosmo.Ogamma(z) + cosmo.Onu(z))  # type: ignore[no-any-return]


@overload
def z_matter_radiation_equality(
    cosmo: FLRW, *, full_output: Literal[False] = ...
) -> Quantity:
    ...


@overload
def z_matter_radiation_equality(
    cosmo: FLRW, *, full_output: Literal[True]
) -> tuple[Quantity, optim.RootResults]:
    ...


@overload
def z_matter_radiation_equality(
    cosmo: FLRW, zmin: TZ, zmax: TZ, *, full_output: Literal[False] = ..., **rootkw: Any
) -> Quantity:
    ...


@overload
def z_matter_radiation_equality(
    cosmo: FLRW, zmin: TZ, zmax: TZ, *, full_output: Literal[True], **rootkw: Any
) -> tuple[Quantity, optim.RootResults]:
    ...


def z_matter_radiation_equality(
    cosmo: FLRW,
    zmin: TZ = 1e3,
    zmax: TZ = 1e4,
    *,
    full_output: bool = False,
    **rootkw: Any,
) -> Quantity | tuple[Quantity, optim.RootResults]:
    """Calculate matter-radiation equality redshift for a given cosmology.

    This works for a cosmology with any number of components.
    If the matter-radiation equality redshift can be calculated analytically,
    that should be used instead of this general function.

    Parameters
    ----------
    cosmo : `~astropy.cosmology.Cosmology`
        Must have methods ``Om(z)``, ``Ogamma(z)``, and ``Onu(z)``.
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
    `scipy.optimize.RootResults`
        Full results of root finder if `full_output` is True
    """
    # residual function
    zeq, rest = optim.brentq(
        _residual, zmin, zmax, args=(cosmo,), full_output=True, **rootkw
    )
    z_eq = zeq << cu.redshift

    return z_eq if not full_output else (z_eq, rest)


# ==================================================================


@dataclass(frozen=True)
class DistanceMeasureConverter:
    """Convert between distance measures."""

    cosmo: FLRW | ProxyType[FLRW]

    # ---------------------------------
    # scales

    @property
    def lambda0(self) -> Quantity:
        """Distance scale factor.

        Returns
        -------
        Quantity
        """
        aeq = self.a_today / (1.0 + self.z_matter_radiation_equality)
        lambda0 = (const.c / self.cosmo.H0) * np.sqrt((8 * aeq) / self.cosmo.Om0)
        return lambda0 << u.Mpc

    # ---------------------------------
    # Beginning of time

    @property
    def z_begin(self) -> Quantity:
        """Redshift at the beginning of time."""
        return np.inf << cu.redshift

    @property
    def a_begin(self) -> Quantity:
        """Scale-factor at the beginning of time."""
        return 0 << u.one

    @property
    def rho_begin(self) -> Quantity:
        """Rho at the beginning of time."""
        return 1 / np.sqrt(2) << u.one

    # ---------------------------------
    # matter-radiation equality

    def calculate_z_matter_radiation_equality(
        self, zmin: TZ = 1e3, zmax: TZ = 1e4, **rootkw: Any
    ) -> Quantity:
        """Redshift at matter-radiation equality.

        Parameters
        ----------
        zmin, zmax : float
            The min/max redshift in which to search for matter-radiation equality.
        **rootkw : Any
            kwargs into scipy minimizer. See `~scipy.optimize.brentq` for details.

        Returns
        -------
        Quantity
            The redshift at matter-radiation equality.
        """
        return z_matter_radiation_equality(
            self.cosmo, zmin=zmin, zmax=zmax, full_output=False, **rootkw
        )

    @cached_property
    def z_matter_radiation_equality(self) -> Quantity:
        """Redshift at matter-radiation equality."""
        return self.calculate_z_matter_radiation_equality()

    @property
    def a_matter_radiation_equality(self) -> Quantity:
        """Scale factor at matter-radiation equality."""
        return self.a_of_z(self.z_matter_radiation_equality)

    @property
    def rho_matter_radiation_equality(self) -> Quantity:
        """Rho at matter-radiation equality."""
        return self.rho_of_z(self.z_matter_radiation_equality)

    # ---------------------------------
    # Today

    @property
    def z_today(self) -> Quantity:
        """Redshift today."""
        return 0.0 * cu.redshift

    @property
    def a_today(self) -> Quantity:
        """Scale factor today."""
        return 1.0 * u.one

    @property
    def rho_today(self) -> Quantity:
        """Rho today."""
        return np.float64("41.16336546526635") << u.one

    # ---------------------------------
    # Redshift and scale factor

    def z_of_a(self, a: Quantity | float, /) -> Quantity:
        """Redshift from the scale factor."""
        return (np.divide(1.0, a) - 1.0) << cu.redshift

    def a_of_z(self, z: Quantity | float, /) -> Quantity:
        """Scale factor from the redshift."""
        return (1.0 / (z + 1.0)) << u.one

    # ---------------------------------
    # Scale factor and rho

    def a_of_rho(self, rho: Quantity | float, /) -> Quantity:
        """Scale factor from rho.

        Clipped to the range [``a_begin``, ``a_today``].
        """
        return (
            np.clip(
                self.a_matter_radiation_equality * (2 * rho**2 - 1),
                self.a_begin,
                self.a_today,
            )
            << u.one
        )

    def rho_of_a(self, a: Quantity | float, /) -> Quantity:
        """rho from the scale factor.

        Clipped to the range [``rho_begin``, ``rho_today``].
        """
        return (
            np.clip(
                np.sqrt((1 + a / self.a_matter_radiation_equality) / 2),
                self.rho_begin,
                self.rho_today,
            )
            << u.one
        )

    # ---------------------------------
    # Redshift and rho

    def z_of_rho(self, rho: Quantity | float, /) -> Quantity:
        """Redshift from rho."""
        return np.clip(self.z_of_a(self.a_of_rho(rho)), 0, None) << cu.redshift

    def rho_of_z(self, z: Quantity | float, /) -> Quantity:
        """Rho of redshift."""
        return self.rho_of_a(self.a_of_z(z))
