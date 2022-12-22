
"""Initiation Tests for :mod:`~cosmic_conchometer.diffusion_distortion.core`."""

__all__ = ["Test_DiffusionDistortionBase"]


##############################################################################
# IMPORTS


import pytest

# PROJECT-SPECIFIC
from .utils import CLASS
from cosmic_conchometer import diffusion_distortion

##############################################################################
# TESTS
##############################################################################

# -------------------------------------------------------------------


class Test_DiffusionDistortionBase:
    """Test `~cosmic_conchometer.diffusion_distortion.core.DiffusionDistortionBase`."""

    _cls = diffusion_distortion.core.DiffusionDistortionBase

    @classmethod
    def setup_class(cls):
        """Setup Class.

        Setup any state specific to the execution of the given class (which
        usually contains tests).

        """
        
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
