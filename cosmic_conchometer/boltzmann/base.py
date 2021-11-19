# -*- coding: utf-8 -*-

"""Transfer Function."""


##############################################################################
# IMPORTS

# STDLIB
import abc
import typing as T

# THIRD PARTY
import numpy as np

# PROJECT-SPECIFIC
from cosmic_conchometer.common import CosmologyDependent

__all__ = ["TransferFunctionBase"]

##############################################################################
# PARAMETERS

N = T.TypeVar("N", float, np.ndarray)  # some numerical input
# TODO! mypy complains about np.generic

##############################################################################


class TransferFunctionBase(CosmologyDependent):
    """A transfer function."""

    @abc.abstractmethod
    def __call__(self) -> N:
        pass
