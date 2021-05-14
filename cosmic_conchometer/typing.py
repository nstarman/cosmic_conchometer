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
]

##############################################################################
# PARAMETERS

NUMPY_VERSION = tuple([int(x) for x in np.version.version.split(".")])


##############################################################################
# CODE

# if NUMPY_VERSION > (1, 20, 0):
#     # THIRD PARTY
#     import numpy.typing as npt
#
#     ArrayLike = npt.ArrayLike
#
# else:
#     ArrayLike = T.Union[float, list, tuple, np.ndarray]
ArrayLike = T.Union[float, np.float, np.complex, np.ndarray]

TArrayLike = T.TypeVar("TArrayLike", float, np.float, np.complex, np.ndarray)


ArrayLike_Callable = T.Callable[
    [ArrayLike],
    ArrayLike,
]

##############################################################################
# END
