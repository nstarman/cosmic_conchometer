# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Set up module."""

##############################################################################
# IMPORTS

from __future__ import absolute_import

# BUILT-IN
import pathlib
import typing as T

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


class _NoOpPBar(object):
    """This class implements the progress bar interface but does nothing.

    This class is from :mod:`~emcee`.

    """

    def __init__(self, *args: T.Any, **kwargs: T.Any) -> None:
        pass

    # /def

    def __enter__(self, *args: T.Any, **kwargs: T.Any) -> T.Any:
        return self

    # /def

    def __exit__(self, *args: T.Any, **kwargs: T.Any) -> T.Any:
        pass

    # /def

    def update(self, count: T.Any) -> T.Any:
        pass

    # /def


# /class

##############################################################################
# END
