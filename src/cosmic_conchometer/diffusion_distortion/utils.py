"""Utilities for :mod:`cosmic_conchometer.diffusion_distortion`."""

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING, overload

# THIRD-PARTY
import numpy as np

if TYPE_CHECKING:
    # THIRD-PARTY
    from numpy.typing import NDArray


__all__: list[str] = []

##############################################################################
# CODE
##############################################################################


@overload
def rho2_of_rho1(
    rho1: float, spll: float, sprp: float, *, maxrho_domain: float
) -> float:
    ...


@overload
def rho2_of_rho1(
    rho1: float | NDArray,
    spll: float | NDArray,
    sprp: float | NDArray,
    *,
    maxrho_domain: float,
) -> NDArray:
    ...


def rho2_of_rho1(
    rho1: float | NDArray,
    spll: float | NDArray,
    sprp: float | NDArray,
    *,
    maxrho_domain: float,
) -> float | NDArray:
    r""":math:`rho_2 = rho_1 - \sqrt{(s_{\|}+rho_1-rho_V)^2 + s_{\perp}^2}`.

    Parameters
    ----------
    rho1 : float
        Rho.
    spll, sprp : float
        S.
    maxrho_domain : float
        Maximum valid rho.

    Returns
    -------
    float
    """
    return rho1 - np.sqrt((spll + rho1 - maxrho_domain) ** 2 + sprp**2)
