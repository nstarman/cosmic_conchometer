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
from classy import Class as CLASS

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
    assert common._Ak_unity(np.random.rand(1) * 1e5) == 1.0 + 0j


# /def

# -------------------------------------------------------------------


class Test_default_Ak:
    def test__default_value(self):
        """check attribute ``_default_value``."""
        assert common.default_Ak._default_value == "unity"

    def test__value(self):
        """check attribute ``_value``."""
        assert common.default_Ak._value is None

    def test__registry(self):
        """check attribute ``_registry``."""
        assert isinstance(common.default_Ak._registry, dict)
        assert common.default_Ak._default_value in common.default_Ak._registry

    def test_get_from_str(self):
        """Test method ``get_from_str``."""
        # success
        assert common.default_Ak.get_from_str("unity") is common._Ak_unity

        # failure
        with pytest.raises(KeyError):
            common.default_Ak.get_from_str("not registered")

    def test_validate(self):
        """Test method ``validate``."""
        # None
        assert common.default_Ak.validate(None) is common._Ak_unity

        # str
        assert common.default_Ak.validate("unity") is common._Ak_unity
        with pytest.raises(KeyError):
            common.default_Ak.validate("not registered")

        # callable
        assert common.default_Ak.validate(common._Ak_unity) is common._Ak_unity

        Akf = lambda x: 2j
        assert common.default_Ak.validate(Akf) is Akf

        # TypeError
        with pytest.raises(TypeError):
            common.default_Ak.validate(object())

    def test_register(self):
        """Test method ``register``."""
        # error
        with pytest.raises(ValueError, match="cannot overwrite"):
            common.default_Ak.register("unity", None, overwrite=False)

        # successful
        try:
            common.default_Ak.register("new", lambda x: 2j, overwrite=False)
        except Exception:
            raise
        else:
            common.default_Ak._registry["new"](10) == 2j
        finally:
            common.default_Ak._registry.pop("new", None)


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

        cls.instance = cls._cls(cls.cosmo)

    # /def

    # =====================================================
    # Method Tests

    def test___init__(self):
        """Test method ``__init__``."""
        # Most basic instantiation.
        inst = self._cls(self.cosmo)

        assert isinstance(inst, self._cls)
        assert isinstance(inst.cosmo, self.cosmo)

        # with metadata
        self._cls(self.cosmo, meta=dict(extra="info"))

        assert isinstance(inst, self._cls)
        assert isinstance(inst.cosmo, self.cosmo)
        assert inst.meta["extra"] == "info"

    # /def

    def test_attributes(self):
        """Test class has expected attributes."""
        # make class
        inst = self._cls(self.cosmo)

        # test has attributes
        for attr in "cosmo":
            hasattr(inst, attr)

    # /def


# /class


##############################################################################
# END
