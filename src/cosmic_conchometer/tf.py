"""Type hints."""

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING

# THIRD-PARTY
import numpy as np
from scipy.constants import speed_of_light

# LOCAL
from cosmic_conchometer.params import LCDMParameters

if TYPE_CHECKING:
    # THIRD-PARTY
    from astropy.units import Quantity
    from numpy.typing import NDArray

__all__ = ["baumann_transfer_function"]

##############################################################################


def baumann_transfer_function(
    cosmo: LCDMParameters,
    kmag: float | NDArray,
    /,
    *,
    z_last_scatter: float | NDArray | Quantity = 1_100.0,
) -> NDArray:
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
    aeq = 1.0 / (1.0 + cosmo.z_matter_radiation_equality)
    lambda0 = (speed_of_light / (100 * cosmo.h)) * np.sqrt((8 * aeq) / cosmo.Om0)

    R = 3e4 * cosmo.Ob0 * cosmo.h**2 / (1 + z_last_scatter)

    alpha_ls = (1 + cosmo.z_matter_radiation_equality) / (1 + z_last_scatter)
    kappa = R / alpha_ls

    sig2LS = (
        lambda0
        / np.sqrt(6 * kappa)
        * (
            np.arcsinh(np.sqrt((1 + alpha_ls) * kappa / (1 - kappa)))
            - np.arcsinh(np.sqrt(kappa / (1 - kappa)))
        )
    )

    # Note the approximation that sigLS = sig2LS
    return -0.2 * ((1 + 3 * R) * np.cos(sig2LS * kmag) - 3 * R)
