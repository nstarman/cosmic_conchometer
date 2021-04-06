# -*- coding: utf-8 -*-

"""Intrinsic Distortion Core Functions."""

__all__ = [
    "IntrinsicDistortionBase",
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
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

# PROJECT-SPECIFIC
from cosmic_conchometer.common import CosmologyDependent, default_Ak

##############################################################################
# PARAMETERS

IUSType = T.Callable[[T.Union[float, np.ndarray]], np.ndarray]

ArrayLike_Callable = T.Callable[
    [T.Union[float, np.ndarray]],
    T.Union[float, np.ndarray],
]


##############################################################################
# CODE
##############################################################################


class IntrinsicDistortionBase(CosmologyDependent):
    r"""Spectral Distortion.

    Parameters
    ----------
    cosmo : `~astropy.cosmology.core.Cosmology` instance
    class_cosmo : :class:`classy.Class`
    AkFunc: Callable or str or None (optional, keyword-only)
        The function to calculate :math:`A(\vec{k})`

    Other Parameters
    ----------------
    integration_method : callable
        The function to perform integrals.

    """

    def __init__(
        self,
        cosmo: Cosmology,
        class_cosmo: Class,
        *,
        AkFunc: T.Union[str, ArrayLike_Callable, None] = None,
    ) -> None:
        super().__init__(cosmo)
        self.class_cosmo = class_cosmo  # TODO maybe move to superclass

        self.AkFunc: ArrayLike_Callable
        if AkFunc is None:
            self.AkFunc = default_Ak.get()
        elif isinstance(AkFunc, str) or callable(AkFunc):
            with default_Ak.set(AkFunc):
                self.AkFunc = default_Ak.get()
        else:
            raise TypeError("AkFunc must be <None, str, callable>.")

        # calculated quantities
        # TODO methods to set zeta array?
        thermo = class_cosmo.get_thermodynamics()
        self._zeta_arr = self.zeta(thermo["z"])
        self._PgamBar_arr = thermo["exp(-kappa)"]
        self._GgamBar_arr = thermo["g [Mpc^-1]"]

        # FIXME! units
        self.PgamBarCL: IUSType = IUS(self._zeta_arr, self._PgamBar_arr)
        self.GgamBarCL: IUSType = IUS(self._zeta_arr, self._GgamBar_arr)

        self.PgamBarCL0: float = self.PgamBarCL(self.zeta0)

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
