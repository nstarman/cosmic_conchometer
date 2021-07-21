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
from classy import Class as CLASS

# PROJECT-SPECIFIC
from cosmic_conchometer.common import CosmologyDependent, default_Ak
from cosmic_conchometer.typing import ArrayLike, ArrayLikeCallable
from cosmic_conchometer.utils import distances

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
        class_cosmo: CLASS,
        *,
        AkFunc: T.Union[str, ArrayLikeCallable, None] = None,
        **kwargs
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

    # /def

    @staticmethod
    def set_AkFunc(value: T.Union[None, str, T.Callable]) -> T.Callable:
        """Set the default function used in A(k).

        Can be used as a contextmanager, same as `~.default_Ak`.

        Parameters
        ----------
        value

        Returns
        -------
        `~astropy.utils.state.ScienceStateContext`
            Output of :meth:`~.default_Ak.set`

        """
        self.Akfunc: T.Callable = default_Ak.set(value)
        return self.Akfunc

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
