"""Initial Conditions."""

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING, TypeVar

# THIRD-PARTY
import numpy as np

# LOCAL
from cosmic_conchometer.params import LCDMParameters

if TYPE_CHECKING:
    # THIRD-PARTY
    from astropy.units import Quantity

__all__: list[str] = []

T = TypeVar("T", np.ndarray, "Quantity")

##############################################################################


def power_spectrum(
    cosmo: LCDMParameters,
    kmag: T,
    /,
    *,
    pivot_scale: T,
) -> T:
    """Simple power spectrum.

    Parameters
    ----------
    cosmo : `~cosmic_conchometer.LCDMParameters`, position-only
        Cosmology parameters.
    kmag : NDArray | Quantity, position-only
        k magnitude.
    pivot_scale : NDArray | Quantity
        pivot scale.
        Must have units compatible with inverse Mpc.

    Returns
    -------
    NDArray | Quantity
    """
    return (
        cosmo.As * np.power(kmag / pivot_scale, cosmo.ns - 1) / (4 * np.pi * kmag**3)
    )
