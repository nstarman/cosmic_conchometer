# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Set up module."""

##############################################################################
# IMPORTS

from __future__ import absolute_import

# BUILT-IN
import pathlib

__all__ = [
    "DATA_DIR",
    # flags,
    "HAS_TQDM",
]


##############################################################################
# PARAMETERS

DATA_DIR: str = str(pathlib.Path(__file__).parent.joinpath("data"))

# -------------------------------------------------------------------

HAS_TQDM: bool
try:
    # THIRD PARTY
    from tqdm import tqdm  # noqa: F401
except ImportError:
    HAS_TQDM = False
else:
    HAS_TQDM = True


##############################################################################
# END
