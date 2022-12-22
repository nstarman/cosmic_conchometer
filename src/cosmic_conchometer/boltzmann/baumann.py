
"""Transfer Function."""

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING, Mapping
from functools import cached_property


import astropy.units as u
import numpy as np

# PROJECT-SPECIFIC
from .base import TransferFunctionBase

if TYPE_CHECKING:
    from astropy.cosmology import Cosmology

__all__ = ["TransferFunction"]

##############################################################################


class TransferFunction(TransferFunctionBase):
    
    def __init__(self, cosmo: Cosmology, redshift_last_scatter: float=1_100,
                 *, meta: Mapping | None = None) -> None:
        super().__init__(cosmo, meta=meta)
        self.z_last_scatter = redshift_last_scatter

    @cached_property
    def R(self) -> float:
        return 3e4 * self._cosmo.Ob0 * self._cosmo.h ** 2 / (1 + self.z_last_scatter)

    @cached_property
    def sig2LS(self):
        alphaLS = (1 + self.z_eq) / (1 + self.z_last_scatter)
        kappa = self.R / alphaLS
        prefactor = self.lambda0.to(u.Mpc) / np.sqrt(6 * kappa)

        return prefactor * (
            + np.arcsinh( np.sqrt( (1 + alphaLS) * kappa / (1 - kappa) )) 
            - np.arcsinh( np.sqrt( kappa / (1 - kappa) )) 
        )

    def __call__(self, k: u.Quantity, /) -> float:
        # Note the approximation that sigLS = sig2LS
        if isinstance(k, u.Quantity):
            arg = (k * self.sig2LS).to_value(u.one)
        else:
            arg = k * self.sig2LS.value

        return -0.2 * ( (1 + 3 * self.R) * np.cos(arg) - 3 * self.R)
