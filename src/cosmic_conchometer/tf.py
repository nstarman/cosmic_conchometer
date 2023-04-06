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

speed_of_light_kms: float = speed_of_light / 1000.0


class TransferFunctionCallable(Protocol):
    """Protocol for transfer function functions."""

    def __call__(
        self,
        params: LCDMParameters,
        kmag: scalarT | NDAf,
        /,
        *,
        z_last_scatter: scalarT,
    ) -> NDAf:
        """Transfer function."""
        ...


##############################################################################


def _baumann_transfer_function_terms(
    p: LCDMParameters, /, *, z_last_scatter: scalarT = 1_100.0
) -> tuple[scalarT, scalarT]:
    """Transfer function from Baumann.

    Parameters
    ----------
    p : `~cosmic_conchometer.params.LCDMParameters`, position-only
        Cosmology parameters.
    kmag : NDArray, position-only
        k magnitude.

    z_last_scatter : float, keyword-only
        Redshift of last scattering.

    Returns
    -------
    scalarT, scalarT
    """
    # lambda0 from distance_measures
    aeq: scalarT = 1.0 / (1.0 + p.z_matter_radiation_equality.value)
    lambda0: scalarT = (speed_of_light_kms / (100 * p.h)) * np.sqrt((8 * aeq) / p.Om0)

    R = 0.75 * (p.Ob0 / p.Ogamma0) / (1 + z_last_scatter)

    alpha_ls: scalarT = (1 + p.z_matter_radiation_equality.value) / (1 + z_last_scatter)
    kappa = R / alpha_ls

    sig2LS: scalarT = (
        lambda0
        / np.sqrt(6 * kappa)
        * (
            np.arcsinh(np.sqrt((1 + alpha_ls) * kappa / (1 - kappa)))
            - np.arcsinh(np.sqrt(kappa / (1 - kappa)))
        )
    )
    return R, sig2LS


def baumann_transfer_function(
    params: LCDMParameters,
    kmag: scalarT | NDAf,
    /,
    *,
    z_last_scatter: scalarT = 1_100.0,
) -> NDAf:
    """Transfer function from Baumann.

    Parameters
    ----------
    params : `~cosmic_conchometer.params.LCDMParameters`, position-only
        Cosmology parameters.
    kmag : NDArray, position-only
        k magnitude.

    z_last_scatter : float, keyword-only
        Redshift of last scattering.

    Returns
    -------
    NDArray
    """
    R, sig2LS = _baumann_transfer_function_terms(params, z_last_scatter=z_last_scatter)
    # Note the approximation that sigLS = sig2LS
    return -0.2 * ((1 + 3 * R) * np.cos(sig2LS * kmag) - 3 * R)
