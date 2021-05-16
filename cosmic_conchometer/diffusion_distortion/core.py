# -*- coding: utf-8 -*-

"""Intrinsic Distortion Core Functions."""

__all__ = [
    "DiffusionDistortionBase",
]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T
from abc import abstractmethod

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.cosmology.core import Cosmology
from classy import Class

# PROJECT-SPECIFIC
from cosmic_conchometer.common import CosmologyDependent, default_Ak
from cosmic_conchometer.typing import ArrayLike, ArrayLikeCallable

##############################################################################
# PARAMETERS

IUSType = T.Callable[[ArrayLike], np.ndarray]


##############################################################################
# CODE
##############################################################################


class DiffusionDistortionBase(CosmologyDependent):
    r"""Spectral Distortion.

    Parameters
    ----------
    cosmo : `~astropy.cosmology.core.Cosmology`
    class_cosmo : :class:`classy.Class`
    AkFunc: Callable, str, or None (optional, keyword-only)
        The function to calculate :math:`A(\vec{k})`

    """

    def __init__(
        self,
        cosmo: Cosmology,
        class_cosmo: Class,
        *,
        AkFunc: T.Union[str, ArrayLikeCallable, None] = None,
    ) -> None:
        super().__init__(cosmo)
        self.class_cosmo = class_cosmo  # TODO? move to superclass

        self.AkFunc: ArrayLikeCallable
        if AkFunc is None:
            self.AkFunc = default_Ak.get()
        elif isinstance(AkFunc, str) or callable(AkFunc):
            with default_Ak.set(AkFunc):
                self.AkFunc = default_Ak.get()
        else:
            raise TypeError("AkFunc must be <None, str, callable>.")

        # # calculated quantities
        # # TODO methods to set zeta array?
        # thermo = class_cosmo.get_thermodynamics()
        # self._zeta_arr = self.zeta(thermo["z"])
        # self._PgamBar_arr = thermo["exp(-kappa)"]
        # self._GgamBar_arr = thermo["g [Mpc^-1]"]

        #
        # # FIXME! units
        # self.PgamBarCL: IUSType = IUS(self._zeta_arr, self._PgamBar_arr)
        # self.GgamBarCL: IUSType = IUS(self._zeta_arr, self._GgamBar_arr)
        #
        # self.PgamBarCL0: float = self.PgamBarCL(self.zeta0)

    # /def

    # ------------------------------

    @abstractmethod
    def __call__(self) -> u.Quantity:
        """Perform computation."""
        pass

    compute = __call__
    # /def

    # ------------------------------


# /class


##############################################################################
# END
