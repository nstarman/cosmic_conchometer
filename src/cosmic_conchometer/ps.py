"""Initial Conditions."""

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING, Protocol

# THIRD-PARTY
import numpy as np

# LOCAL
from cosmic_conchometer.params import LCDMParameters

if TYPE_CHECKING:
    # THIRD-PARTY

    # LOCAL
    from cosmic_conchometer._typing import NDAf, scalarT

__all__ = ["power_spectrum"]

##############################################################################


class PowerSpectrumCallable(Protocol):
    """Protocol for power spectrum functions."""

    def __call__(
        self,
        cosmo: LCDMParameters,
        kmag: scalarT | NDAf,
        /,
        *,
        pivot_scale: scalarT | NDAf,
    ) -> NDAf:
        """Power spectrum function."""
        ...


##############################################################################


def power_spectrum(
    cosmo: LCDMParameters, kmag: scalarT | NDAf, /, *, pivot_scale: scalarT | NDAf
) -> NDAf:
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
