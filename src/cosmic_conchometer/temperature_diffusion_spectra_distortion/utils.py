"""Utilities for :mod:`cosmic_conchometer.diffusion_distortion`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cosmic_conchometer._typing import NDAf, scalarT


__all__: list[str] = []

##############################################################################


def rho2_of_rho1(
    rho1: scalarT | NDAf,
    spll: scalarT | NDAf,
    sprp: scalarT | NDAf,
    *,
    maxrho_domain: scalarT,
) -> scalarT | NDAf:
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
