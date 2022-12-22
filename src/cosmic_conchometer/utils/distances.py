"""Convert between different distance measures."""

##############################################################################
# IMPORTS

from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property

# STDLIB
from typing import TypeVar, Any, Union, cast


import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.cosmology.core import Cosmology

# PROJECT-SPECIFIC
from cosmic_conchometer.typing import TArrayLike

__all__ = [
    "z_matter_radiation_equality",
    "DistanceMeasureConverter",
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
TZ = TypeVar("TZ", float, u.Quantity)  # u.Quantity[dimensionless]

##############################################################################
# CODE
##############################################################################


def z_matter_radiation_equality(
    cosmo: Cosmology,
    zmin: TZ = 1e3,
    zmax: TZ = 1e4,
    *,
    full_output: bool = False,
    **rootkw: Any,
) -> Union[u.Quantity, tuple[u.Quantity, tuple[Any, ...]]]:
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
    
    from scipy.optimize import brentq

    # residual function
    def f(z: TZ) -> TZ:
        diff: TZ = cosmo.Om(z) - (cosmo.Ogamma(z) + cosmo.Onu(z))
        return diff

    zeq, *rest = brentq(f, zmin, zmax, full_output=True, **rootkw)
    z_eq = cast("u.Quantity", zeq)

    return z_eq if not full_output else (z_eq, rest)


@dataclass(frozen=True)
class DistanceMeasureConverter:

    cosmo: Cosmology

    # ---------------------------------
    # scales

    @property
    def lambda0(self) -> u.Quantity:
        aeq = 1.0 / (1.0 + self.z_matter_radiation_equality)  # a0 = 1
        lambda0 = (const.c / self.cosmo.H0) * np.sqrt(
            (8 * aeq) / self.cosmo.Om0
        )
        return lambda0 << u.Mpc

    # ---------------------------------
    # beginning of time

    @property
    def z_begin(self) -> float:
        """Redshift at the beginning of time."""
        return np.inf

    @property
    def a_begin(self) -> int:
        """Redshift at the beginning of time."""
        return 0

    # @property
    # def alpha_begin(self) -> int:
    #     return 0

    @property
    def rho_begin(self) -> float:
        return 1 / np.sqrt(2)

    # ---------------------------------
    # matter-radiation equality

    def calculate_z_matter_radiation_equality(
        self,
        zmin: TZ = 1e3,
        zmax: TZ = 1e4,
        *,
        full_output: bool = False,
        **rootkw: Any,
    ) -> float:
        return z_matter_radiation_equality(
            self.cosmo, zmin=zmin, zmax=zmax, full_output=full_output, **rootkw
        )

    @cached_property
    def z_matter_radiation_equality(self) -> float:
        return self.calculate_z_matter_radiation_equality()

    # def calculate_a_matter_radiation_equality(
    #     self,
    #     amin: float = 1e-9,
    #     amax: float = 0.1,
    #     **rootkw: Any,
    # ) -> float:
    #     """Scale factor at matter-radiation equality."""
    #     zeq = self.calculate_z_matter_radiation_equality(
    #         zmin=self.z_of_a(amin), zmax=self.z_of_a(amax), **rootkw
    #     )
    #     aeq = self.a_of_z(zeq)
    #     return aeq

    @property
    def a_matter_radiation_equality(self) -> float:
        # return self.calculate_a_matter_radiation_equality()
        return self.a_of_z(self.z_matter_radiation_equality)

    # @property
    # def alpha_matter_radiation_equality(self) -> float:
    #     return 1.0

    @property
    def rho_matter_radiation_equality(self) -> float:
        # TODO! have a calculate_rho_matter_radiation_equality
        return self.rho_of_z(self.z_matter_radiation_equality)

    # ---------------------------------
    # today

    @property
    def z_today(self) -> float:
        """Redshift today."""
        return 0.0

    @property
    def a_today(self) -> float:
        """Scale factor today."""
        return 1.0

    # @property
    # def alpha_today(self) -> float:
    #     """alpha today."""
    #     return 1.0 / self.a_matter_radiation_equality

    # ---------------------------------
    # redshift and scale factor

    def z_of_a(self, a: TArrayLike, /) -> TArrayLike:
        """Redshift from the scale factor."""
        return np.divide(1.0, a) - 1.0

    def a_of_z(self, z: TArrayLike, /) -> TArrayLike:
        """Scale factor from the redshift."""
        return 1.0 / (z + 1.0)

    # # ---------------------------------
    # # scale factor and alpha

    # def a_of_alpha(self, alpha: TArrayLike, /) -> TArrayLike:
    #     """Scale factor from alpha.

    #     :math:`\alpha = a / a_{eq}`
    #     """
    #     a: TArrayLike = self.a_matter_radiation_equality * alpha
    #     return a

    # def alpha_of_a(self, a: TArrayLike, /) -> TArrayLike:
    #     """alpha from the scale factor."""
    #     alpha: TArrayLike = a / self.a_matter_radiation_equality
    #     return alpha

    # # ---------------------------------
    # # redshift and alpha

    # def z_of_alpha(self, alpha: TArrayLike, /) -> TArrayLike:
    #     """redshift from alpha."""
    #     return self.z_of_a(self.a_of_alpha(alpha))

    # def alpha_of_z(self, z: TArrayLike, /) -> TArrayLike:
    #     """alpha from redshift."""
    #     return self.alpha_of_a(self.a_of_z(z))

    # # ---------------------------------
    # # alpha and rho

    # def rho_of_alpha(self, alpha: TArrayLike, /) -> TArrayLike:
    #     """rho from alpha."""
    #     rho: TArrayLike = np.sqrt((1 + alpha) / 2.0)
    #     return rho

    # def alpha_of_rho(self, rho: TArrayLike, /) -> TArrayLike:
    #     """alpha from rho."""
    #     alpha: TArrayLike = 2 * rho ** 2 - 1
    #     return alpha

    # ---------------------------------
    # scale factor and rho

    def a_of_rho(self, rho: TArrayLike, /) -> TArrayLike:
        """"""
        return self.a_matter_radiation_equality * (2 * rho ** 2 - 1)

    def rho_of_a(self, a: TArrayLike, /) -> TArrayLike:
        return np.sqrt((1 + a / self.a_matter_radiation_equality) / 2)

    # ---------------------------------
    # redshift and rho

    def z_of_rho(self, rho: TArrayLike, /) -> TArrayLike:
        return self.z_of_a(self.a_of_rho(rho))

    def rho_of_z(self, z: TArrayLike, /) -> TArrayLike:
        return self.rho_of_a(self.a_of_z(z))
