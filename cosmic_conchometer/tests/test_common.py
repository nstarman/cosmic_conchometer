# -*- coding: utf-8 -*-

"""Initiation Tests for :mod:`~cosmic_conchometer.common`."""

__all__ = ["Test_DiffusionDistortionBase"]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.units as u
import numpy as np
import pytest
from astropy.cosmology import Cosmology, default_cosmology
from classy import CLASS

# PROJECT-SPECIFIC
from cosmic_conchometer import common
from cosmic_conchometer.config import conf

##############################################################################
# TESTS
##############################################################################


def test_default_cosmo():
    """Test `~cosmic_conchometer.common.default_cosmo`"""
    assert isinstance(common.default_cosmo, Cosmology)

    with default_cosmology.set(conf.default_cosmo):
        assert common.default_cosmo == default_cosmology.get()


# /def

# -------------------------------------------------------------------


def test_blackbody():
    """Test :func:`~cosmic_conchometer.common.blackbody`"""
    # standard usage
    bb = common.blackbody(10 * u.GHz, 1e5 * u.K)
    assert bb.unit == common._bb_unit

    # some values
    assert np.isnan(common.blackbody(0 * u.GHz, 1e5 * u.K))

    assert common.blackbody(10 * u.GHz, 0 * u.K) == 0


# /def

# -------------------------------------------------------------------


def test__Ak_unity():
    """Test :func:`~cosmic_conchometer.common._Ak_unity`"""
    assert common._Ak_unity() == 1.0 + 0j


# -------------------------------------------------------------------


class Test_CosmologyDependent:
    """Test `~cosmic_conchometer.common.CosmologyDependent`."""

    _cls = common.CosmologyDependent

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

    # =====================================================
    # Method Tests

    def test___init__(self):
        """Test method ``__init__``."""
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
        inst = self._cls(self.cosmo, self.class_cosmo)

        # test has attributes
        # defer tests of CosmologyDependent variables for that test suite.
        # except "cosmo"
        for attr in ("cosmo", "class_cosmo", "AkFunc"):
            hasattr(inst, attr)

    # /def


# /class


##############################################################################
# END
