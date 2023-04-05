"""Tests for core."""


from collections.abc import Mapping
from types import MappingProxyType

import astropy.cosmology.units as cu
import astropy.units as u
import numpy as np
import pytest
from cosmic_conchometer.temperature_diffusion_spectra_distortion.core import (
    SpectralDistortion,
)
from cosmic_conchometer.temperature_diffusion_spectra_distortion.prob_2ls import (
    ComputePspllSprp,
)

from ..test_common import Test_CosmologyDependent
from .utils import CLASS

##############################################################################
# TESTS
##############################################################################


class Test_SpectralDistortion(Test_CosmologyDependent):
    """Test `~cosmic_conchometer.common.CosmologyDependent`."""

    @pytest.fixture(scope="class")
    def cdep_cls(self):
        """Fixture for the cosmology class."""
        return SpectralDistortion

    @pytest.fixture(scope="class")
    def class_cosmo(self):
        """Fixture for the class class."""
        return CLASS()

    @pytest.fixture(scope="class")
    def cdep(self, cdep_cls, cosmo, class_cosmo):
        """Fixture for the cosmology-dependent instance."""
        return cdep_cls(cosmo, class_cosmo)

    # =====================================================
    # Method Tests

    def test___init__(self, cdep_cls, cosmo, class_cosmo):
        """Test method ``__init__``."""
        with pytest.raises(TypeError, match="SpectralDistortion"):
            super().test___init__(cdep_cls, cosmo)

        # Most basic instantiation.
        inst = cdep_cls(cosmo, class_cosmo)

        assert isinstance(inst, cdep_cls)
        assert inst.cosmo is cosmo

        # TODO: more

    def test__class_thermo(self, cdep):
        """Test method ``_class_thermo``."""
        assert isinstance(cdep._class_thermo, Mapping)

        # All values are NDArrays and not writable.
        for val in cdep._class_thermo.values():
            assert isinstance(val, np.ndarray)
            assert not val.flags.writeable

        # known keys
        assert "z" in cdep._class_thermo
        assert cdep._class_thermo["z"].unit == cu.redshift

        assert "g" in cdep._class_thermo
        assert cdep._class_thermo["g"].unit == 1 / u.Mpc

        assert "rho" in cdep._class_thermo
        assert cdep._class_thermo["rho"].unit == u.one

        assert cdep._class_thermo["rho"].value == pytest.approx(
            cdep.distance_converter.rho_of_z(cdep._class_thermo["z"]).value,
        )

    def test_class_thermo(self, cdep):
        """Test attribute ``class_thermo``."""
        # Not in __dict__
        assert "class_thermo" not in cdep.__dict__

        # Read-only
        assert cdep.class_thermo == cdep._class_thermo
        assert isinstance(cdep.class_thermo, MappingProxyType)

        # Now in __dict__
        assert "class_thermo" in cdep.__dict__

    def test_z_domain(self, cdep):
        """Test attribute ``rho_domain``."""
        assert cdep.z_domain == (cdep._class_thermo["z"][0], 100)

    def test_P(self, cdep):
        """Test method ``P``."""
        # Check that the attribute is a DistanceMeasureConverter.
        assert isinstance(cdep.P, ComputePspllSprp)

        # The rest of the tests are in test_prob_2ls

    def test_rho_domain(self, cdep):
        """Test attribute ``rho_domain``."""
        # Obvious
        assert cdep.rho_domain == (
            cdep.distance_converter.rho_of_z(cdep.z_domain[0]),
            cdep.distance_converter.rho_of_z(cdep.z_domain[1]),
        )

        # Slightly more manual
        assert cdep.rho_domain == (
            cdep.distance_converter.rho_of_z(cdep._class_thermo["z"][0]),
            cdep.distance_converter.rho_of_z(100),
        )

    def test_maxrho_domain(self, cdep):
        """Test attribute ``maxrho_domain``."""
        assert cdep.maxrho_domain == cdep.rho_domain[1]
        assert cdep.maxrho_domain == max(cdep.rho_domain)

    def test_z_recombination(self, cdep):
        """Test attribute ``z_recombination``."""
        # Exact
        assert (
            cdep.z_recombination.value
            == cdep._class_thermo["z"][cdep._class_thermo["g"].argmax()]
        )
        assert cdep.z_recombination.unit == cu.redshift

        # Hard-coded
        assert cdep.z_recombination.value == pytest.approx(1089.0, rel=0.2)

    def test_rho_recombination(self, cdep):
        """Test attribute ``rho_recombination``."""
        # Exact
        assert (
            cdep.rho_recombination.value
            == cdep._class_thermo["rho"][cdep._class_thermo["g"].argmax()]
        )
        assert cdep.rho_recombination.unit == u.one

        # Hard-coded
        assert cdep.rho_recombination.value == pytest.approx(1.4333, rel=0.001)

    # =====================================================
    # plots

    @pytest.mark.mpl_image_compare()
    def test_plot_CLASS_points_distribution(self, cdep):
        """Test method ``plot_CLASS_points_distribution``."""
        return cdep.plot_CLASS_points_distribution()

    @pytest.mark.mpl_image_compare()
    def test_plot_zv_choice(self, cdep):
        """Test method ``plot_zv_choice``."""
        return cdep.plot_zv_choice()
