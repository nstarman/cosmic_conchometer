# -*- coding: utf-8 -*-

"""Initiation Tests for `~cosmic_conchometer`."""

__all__ = ["test__ArrayLike_Callable", "Test_IntrinsicDistortionBase"]


##############################################################################
# IMPORTS

# BUILT-IN
import collections.abc as cabc
import typing as T

# THIRD PARTY
import numpy as np
import pytest

# PROJECT-SPECIFIC
from .utils import CLASS
from cosmic_conchometer import diffusion_distortion

##############################################################################
# TESTS
##############################################################################


def test__ArrayLike_Callable():
    """Test `~cosmic_conchometer.diffusion_distortion.core._ArrayLike_Callable`."""
    AC = diffusion_distortion.core._ArrayLike_Callable

    assert T.get_origin(AC) == cabc.Callable

    args = T.get_args(AC)

    # input
    assert isinstance(args[0], list)
    assert T.get_origin(args[0][0]) == T.Union
    assert T.get_args(args[0][0]) == (float, np.ndarray)
    # output
    assert T.get_origin(args[1]) == T.Union
    assert T.get_args(args[1]) == (float, np.ndarray)


# /def


# -------------------------------------------------------------------


class Test_IntrinsicDistortionBase:
    """Test `~cosmic_conchometer.diffusion_distortion.core.IntrinsicDistortionBase`."""

    _cls = diffusion_distortion.core.IntrinsicDistortionBase

    @classmethod
    def setup_class(cls):
        """Setup Class.

        Setup any state specific to the execution of the given class (which
        usually contains tests).

        """
        # THIRD PARTY
        from astropy.cosmology import Planck15

        # input
        cls.cosmo = Planck15
        cls.class_cosmo = CLASS()

        cls.instance = cls._cls(cls.cosmo, cls.class_cosmo, AkFunc="unity")

    # /def

    def test_instantiation(self):
        """Test class instantiation options, including exceptions."""
        # Most basic instantiation.
        self._cls(self.cosmo, self.class_cosmo)

        # Specifying values.
        options_AkFunc = (None, "unity", lambda x: x)
        for value in options_AkFunc:
            self._cls(self.cosmo, self.class_cosmo, AkFunc=value)
        with pytest.raises(TypeError):
            self._cls(self.cosmo, self.class_cosmo, AkFunc=TypeError())

    # /def

    def test_attributes(self):
        """Test class has expected attributes."""
        # make class
        idb = self._cls(self.cosmo, self.class_cosmo)

        # test has attributes
        # defer tests of CosmologyDependent variables for that test suite.
        # except "cosmo"
        for attr in ("cosmo", "class_cosmo", "AkFunc", "_zeta_arr"):
            hasattr(idb, attr)

    # /def


# /class


##############################################################################
# END
