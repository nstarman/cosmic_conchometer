"""Tests for :mod:`~cosmic_conchometer.common`."""

# STDLIB
from weakref import ProxyType

# THIRD-PARTY
import pytest
from astropy.cosmology import Planck18

# LOCAL
from cosmic_conchometer.common import CosmologyDependent
from cosmic_conchometer.utils.distances import DistanceMeasureConverter

##############################################################################
# TESTS
##############################################################################


class Test_CosmologyDependent:
    """Test `~cosmic_conchometer.common.CosmologyDependent`."""

    @pytest.fixture(scope="class")
    def cdep_cls(self):
        """Fixture for the cosmology class."""
        return CosmologyDependent

    @pytest.fixture(scope="class")
    def cosmo_cls(self):
        """Fixture for the cosmology class."""
        return type(Planck18)

    @pytest.fixture(scope="class")
    def cosmo(self):
        """Fixture for the cosmology class."""
        return Planck18

    @pytest.fixture(scope="class")
    def cdep(self, cdep_cls, cosmo):
        """Fixture for the cosmology-dependent instance."""
        return cdep_cls(cosmo)

    # =====================================================
    # Method Tests

    def test___init__(self, cdep_cls, cosmo):
        """Test method ``__init__``."""
        # Most basic instantiation.
        inst = cdep_cls(cosmo)

        assert isinstance(inst, cdep_cls)
        assert inst.cosmo is cosmo

    def test_distance_converter(self, cdep):
        """Test method ``distance_converter``."""
        # Check that the attribute is a DistanceMeasureConverter.
        assert isinstance(cdep.distance_converter, DistanceMeasureConverter)

        # Check that the attribute is a proxy to the cosmology.
        assert isinstance(cdep.distance_converter.cosmo, ProxyType)
        assert cdep.distance_converter.cosmo == cdep.cosmo

        # Further tests are in tests/utils/test_distances.py!

    def test_lambda0(self, cdep):
        """Test method ``lambda0``."""
        # Check the unit.
        assert cdep.lambda0.unit == "Mpc"

        # Check the value is from the distance_converter.
        assert cdep.lambda0 == cdep.distance_converter.lambda0

    def test_z_matter_radiation_equality(self, cdep):
        """Test method ``z_matter_radiation_equality``."""
        # Check the value is from the distance_converter.
        assert (
            cdep.z_matter_radiation_equality
            is cdep.distance_converter.z_matter_radiation_equality
        )
