"""Tests for :mod:`~cosmic_conchometer.params`."""

# THIRD-PARTY
import pytest
from astropy.cosmology import LambdaCDM, Planck18

# LOCAL
from cosmic_conchometer.params import CosmologyParameters, LCDMParameters, planck18
from cosmic_conchometer.utils.distances import z_matter_radiation_equality

##############################################################################
# TESTS
##############################################################################


class Test_CosmologyParameters:
    """Test :class:`~cosmic_conchometer.params.CosmologyParameters`."""

    # ========================================================================
    # Fixtures

    @pytest.fixture(scope="class")
    def params_cls(self):
        """Fixture for the cosmology parameters' class."""
        return CosmologyParameters

    @pytest.fixture(scope="class")
    def params(self, params_cls):
        """Fixture for the cosmology parameters."""
        return params_cls()

    # ========================================================================
    # Method Tests

    def test_initialization(self, params_cls):
        """Test initialization."""
        params_cls()


##############################################################################


class Test_LCDMParameters(Test_CosmologyParameters):
    """Test :class:`~cosmic_conchometer.params.LCDMParameters`."""

    # ========================================================================
    # Fixtures

    @pytest.fixture(scope="class")
    def params_cls(self):
        """Fixture for the cosmology parameters' class."""
        return LCDMParameters

    @pytest.fixture(scope="class")
    def params_params(self):
        """Fixture for the params fields."""
        return {
            "h": 0.70,
            "Om0": 0.30,
            "Ode0": 0.70,
            "Tcmb0": 2.715,
            "Neff": 3.05,
            "Ob0": 0.05,
            "As": 1,
            "ns": 0.965,
        }

    @pytest.fixture(scope="class")
    def params(self, params_cls, params_params):
        """Fixture for the cosmology parameters."""
        return params_cls(**params_params)

    # ========================================================================
    # Method Tests

    def test_initialization(self, params_cls, params_params):
        """Test initialization."""
        # Most basic instantiation.
        with pytest.raises(TypeError):
            params_cls()  # no parameters given

        # The rest are ini the params fixture.

    def test_attributes(self, params):
        """Test the attributes."""
        # Check that the attributes are set.
        assert params.h == 0.70
        assert params.Om0 == 0.30
        assert params.Ode0 == 0.70
        assert params.Tcmb0 == 2.715
        assert params.Neff == 3.05
        assert params.Ob0 == 0.05
        assert params.As == 1
        assert params.ns == 0.965

    def test_cosmo(self, params):
        """Test the ``cosmo`` attribute."""
        assert isinstance(params.cosmo, LambdaCDM)

    def test_from_astropy(self, params_cls, params):
        """Test the ``from_astropy`` method."""
        # Check that the classmethod works.
        params_cls.from_astropy(params.cosmo, As=1, ns=0.965)

    def test_z_matter_radiation_equality(self, params):
        """Test the equality of the matter-radiation equality redshift."""
        # Check that the equality is correct.
        assert params.z_matter_radiation_equality == z_matter_radiation_equality(
            params.cosmo
        )


##############################################################################


def test_planck18():
    """Test the Planck18 cosmology parameters."""
    assert isinstance(planck18, LCDMParameters)

    assert planck18.cosmo == Planck18
