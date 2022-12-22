# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Typing."""

##############################################################################
# IMPORTS

# STDLIB
import typing as T


import numpy as np

__all__ = [
    "ArrayLike",
    "TArrayLike",
    "ArrayLikeCallable",
]


##############################################################################
# CODE

ArrayLike = T.Union[float, np.number, np.ndarray]

TArrayLike = T.TypeVar("TArrayLike", float, np.number, np.ndarray)

ArrayLikeCallable = T.Callable[
    [ArrayLike],
    ArrayLike,
]

##############################################################################
# END
