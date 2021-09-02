# -*- coding: utf-8 -*-

"""Initiation Tests for :mod:`~cosmic_conchometer.typing`."""

__all__ = [
    "test_ArrayLikeCallable",
]

##############################################################################
# IMPORTS

# STDLIB
import collections.abc as cabc
import typing as T

# THIRD PARTY
import numpy as np

# PROJECT-SPECIFIC
from cosmic_conchometer import typing

##############################################################################
# TESTS
##############################################################################


def test_ArrayLike():
    """Test `~cosmic_conchometer.typing._ArrayLike`."""
    # the type
    assert T.get_origin(typing.ArrayLike) == T.Union

    # contents
    args = T.get_args(typing.ArrayLike)
    assert set(args) == {float, np.float, np.complex, np.ndarray}


# /def

# -------------------------------------------------------------------


def test_TArrayLike():
    """Test `~cosmic_conchometer.typing._TArrayLike`."""
    # the type
    assert typing.TArrayLike.__class__ == T.TypeVar

    # contents
    assert set(typing.TArrayLike.__constraints__) == set(T.get_args(typing.ArrayLike))


# /def

# -------------------------------------------------------------------


def test_ArrayLikeCallable():
    """Test `~cosmic_conchometer.typing._ArrayLikeCallable`."""
    # the type
    assert T.get_origin(typing.ArrayLikeCallable) == cabc.Callable

    # contents
    args = T.get_args(typing.ArrayLikeCallable)

    # input
    assert isinstance(args[0], list)
    assert T.get_origin(args[0][0]) == T.Union
    assert T.get_args(args[0][0]) == T.get_args(typing.ArrayLike)
    # output
    assert T.get_origin(args[1]) == T.Union
    assert T.get_args(args[1]) == T.get_args(typing.ArrayLike)


# /def

##############################################################################
# END
