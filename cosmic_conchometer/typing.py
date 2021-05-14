# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Typing."""

##############################################################################
# IMPORTS

import typing as T
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

if NUMPY_VERSION > (1, 20, 0):
    import numpy.typing as npt

    ArrayLike = npt.ArrayLike

else:
    ArrayLike = T.Union[float, list, tuple, np.ndarray]


TArrayLike = T.TypeVar("TArrayLike", *ArrayLike.__args__)
"""ArrayLike, but output type is input type."""


ArrayLike_Callable = T.Callable[
    [ArrayLike],
    ArrayLike,
]

##############################################################################
# END
