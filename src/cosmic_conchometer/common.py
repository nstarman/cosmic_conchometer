"""Core functions safe to be used in lower modules."""

from __future__ import annotations

# STDLIB
from abc import ABCMeta
from dataclasses import dataclass
from typing import TYPE_CHECKING
from weakref import proxy

# THIRD-PARTY
import astropy.units as u
from astropy.cosmology import FLRW
from astropy.utils.metadata import MetaData

# LOCAL
from cosmic_conchometer.utils.distances import DistanceMeasureConverter

if TYPE_CHECKING:
    # THIRD-PARTY
    from astropy.units import Quantity

__all__ = ["CosmologyDependent"]

##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class CosmologyDependent(metaclass=ABCMeta):
    """Class for coordinating cosmology-dependent calculations.

    Parameters
    ----------
    cosmo : `~astropy.cosmology.core.Cosmology`
        The astropy cosmology.

    Notes
    -----
    This class is intended to be used as a base class for other classes that
    need to perform calculations that depend on an Astropy cosmology.
    """

    cosmo: FLRW

    meta = MetaData()

    def __post_init__(self) -> None:
        self.distance_converter: DistanceMeasureConverter
        object.__setattr__(
            self, "distance_converter", DistanceMeasureConverter(proxy(self.cosmo))
        )

    # ================================================================
    # Properties

    @property
    def lambda0(self) -> Quantity:
        """Distance scale factor in Mpc."""
        return self.distance_converter.lambda0.to(u.Mpc)

    @property
    def z_matter_radiation_equality(self) -> Quantity:
        """Redshift of matter-radiation equality."""
        return self.distance_converter.z_matter_radiation_equality
