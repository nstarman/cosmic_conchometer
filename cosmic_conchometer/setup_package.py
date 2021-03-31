# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Set up module."""

##############################################################################
# IMPORTS

from __future__ import absolute_import

__all__ = ["HAS_TQDM"]


##############################################################################
# PARAMETERS

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
