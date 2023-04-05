"""Type hints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
from scipy.constants import speed_of_light

if TYPE_CHECKING:
    from cosmic_conchometer._typing import NDAf, scalarT
    from cosmic_conchometer.params import LCDMParameters

__all__ = ["baumann_transfer_function"]


##############################################################################


class TransferFunctionCallable(Protocol):
    """Protocol for transfer function functions."""

    def __call__(
        self,
        cosmo: LCDMParameters,
        kmag: scalarT | NDAf,
        /,
        *,
        z_last_scatter: scalarT,
    ) -> NDAf:
        """Transfer function."""
        ...


##############################################################################


def baumann_transfer_function(
    cosmo: LCDMParameters,
    kmag: scalarT | NDAf,
    /,
    *,
    z_last_scatter: scalarT = 1_100.0,
) -> NDAf:
    """Transfer function from Baumann.

    Parameters
    ----------
    cosmo : `~cosmic_conchometer.params.LCDMParameters`, position-only
        Cosmology parameters.
    kmag : NDArray, position-only
        k magnitude.

    z_last_scatter : float, keyword-only
        Redshift of last scattering.

    Returns
    -------
    NDArray
    """
    # lambda0 from distance_measures
    aeq: scalarT = 1.0 / (1.0 + cosmo.z_matter_radiation_equality.value)
    lambda0: scalarT = (speed_of_light / 1000 / (100 * cosmo.h)) * np.sqrt(
        (8 * aeq) / cosmo.Om0,
    )

    R = 0.75 * (cosmo.Ob0 / cosmo.Ogamma0) / (1 + z_last_scatter)

    alpha_ls: scalarT = (1 + cosmo.z_matter_radiation_equality.value) / (
        1 + z_last_scatter
    )
    kappa = R / alpha_ls

    sig2LS: scalarT = (
        lambda0
        / np.sqrt(6 * kappa)
        * (
            np.arcsinh(np.sqrt((1 + alpha_ls) * kappa / (1 - kappa)))
            - np.arcsinh(np.sqrt(kappa / (1 - kappa)))
        )
    )

    # Note the approximation that sigLS = sig2LS
    return -0.2 * ((1 + 3 * R) * np.cos(sig2LS * kmag) - 3 * R)
