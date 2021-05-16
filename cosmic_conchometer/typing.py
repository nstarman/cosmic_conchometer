# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Typing."""

##############################################################################
# IMPORTS

# BUILT-IN
import typing as T

# THIRD PARTY
import numpy as np

__all__ = [
    "ArrayLike",
    "TArrayLike",
    "ArrayLikeCallable",
]


##############################################################################
# CODE

ArrayLike = T.Union[float, np.float, np.complex, np.ndarray]

TArrayLike = T.TypeVar("TArrayLike", float, np.float, np.complex, np.ndarray)

ArrayLikeCallable = T.Callable[
    [ArrayLike],
    ArrayLike,
]

##############################################################################
# END
