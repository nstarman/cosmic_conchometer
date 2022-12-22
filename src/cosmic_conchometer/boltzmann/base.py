
"""Transfer Function."""


##############################################################################
# IMPORTS

# STDLIB
import abc
import typing as T


import astropy.units as u
import numpy as np

# PROJECT-SPECIFIC
from cosmic_conchometer.common import CosmologyDependent
from cosmic_conchometer.typing import ArrayLike

__all__ = ["TransferFunctionBase"]

##############################################################################


class TransferFunctionBase(CosmologyDependent):
    """A transfer function."""

    @abc.abstractmethod
    def __call__(self, k: u.Quantity) -> ArrayLike:
        pass
