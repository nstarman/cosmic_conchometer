"""Tests for :mod:`~cosmic_conchometer.utils.distances`."""

# STDLIB
from math import inf, sqrt
from weakref import proxy

# THIRD-PARTY
import astropy.cosmology.units as cu
import astropy.units as u
import pytest
from astropy.cosmology import Planck18
from pytest import approx
from scipy.optimize import RootResults

# LOCAL
from cosmic_conchometer.utils.distances import (
    DistanceMeasureConverter,
    z_matter_radiation_equality,
)

##############################################################################
# TESTS
##############################################################################


def test_z_matter_radiation_equality():
    """Test `~cosmic_conchometer.utils.distances.z_matter_radiation_equality`."""
    # Basic case
    got = z_matter_radiation_equality(Planck18)
    assert got.value == approx(3386.8453)
    assert got.unit == cu.redshift

    # Full output
    got, res = z_matter_radiation_equality(Planck18, full_output=True)
    assert got.value == approx(3386.8453)
    assert isinstance(res, RootResults)

    # Out of bounds
    with pytest.raises(ValueError):
        z_matter_radiation_equality(Planck18, zmin=0, zmax=1e2)


class Test_DistanceMeasureConverter:

    # ========================================================================
    # Fixtures

    @pytest.fixture(scope="class")
    def converter_cls(self):
        """`~cosmic_conchometer.utils.distances.DistanceMeasureConverter`."""
        return DistanceMeasureConverter

    @pytest.fixture(scope="class")
    def cosmo(self):
        """`~astropy.cosmology.Planck18`."""
        return Planck18

    @pytest.fixture(scope="class")
    def cosmo_proxy(self):
        """`~astropy.cosmology.Planck18`."""
        return proxy(Planck18)

    @pytest.fixture(scope="class")
    def converter(self, converter_cls, cosmo):
        """`~cosmic_conchometer.utils.distances.DistanceMeasureConverter`."""
        return converter_cls(cosmo)

    # ========================================================================

    def test_lambda0(self, converter):
        """Test ``DistanceMeasureConverter.lambda0``."""
        assert converter.lambda0 > 0
        assert converter.lambda0.unit == u.Mpc

    def test_z_begin(self, converter):
        """Test ``DistanceMeasureConverter.z_begin``."""
        assert converter.z_begin == inf
        assert converter.z_begin.unit == cu.redshift

    def test_a_begin(self, converter):
        """Test ``DistanceMeasureConverter.a_begin``."""
        assert converter.a_begin == 0
        assert converter.a_begin.unit == u.dimensionless_unscaled

    def test_rho_begin(self, converter):
        """Test ``DistanceMeasureConverter.rho_begin``."""
        assert converter.rho_begin == 1 / sqrt(2)
        assert converter.rho_begin.unit == u.dimensionless_unscaled

    def test_calculate_z_matter_radiation_equality(self, converter, cosmo):
        """Test ``DistanceMeasureConverter.calculate_z_matter_radiation_equality``."""
        assert (
            converter.calculate_z_matter_radiation_equality()
            == z_matter_radiation_equality(cosmo)
        )

        # TODO: tests with zmin, zmax

    def test_z_matter_radiation_equality(self, converter):
        """Test ``DistanceMeasureConverter.z_matter_radiation_equality``."""
        assert (
            converter.z_matter_radiation_equality
            == converter.calculate_z_matter_radiation_equality()
        )

    def test_a_matter_radiation_equality(self, converter, cosmo):
        """Test ``DistanceMeasureConverter.a_matter_radiation_equality``."""
        # Obvious
        assert converter.a_matter_radiation_equality == converter.a_of_z(
            converter.z_matter_radiation_equality
        )

        # Consistency
        assert converter.a_matter_radiation_equality == cosmo.scale_factor(
            converter.z_matter_radiation_equality
        )

    def test_rho_matter_radiation_equality(self, converter, cosmo):
        """Test ``DistanceMeasureConverter.rho_matter_radiation_equality``."""
        # Obvious
        assert converter.rho_matter_radiation_equality == converter.rho_of_z(
            converter.z_matter_radiation_equality
        )

    def test_z_today(self, converter):
        """Test ``DistanceMeasureConverter.z_today``."""
        assert converter.z_today == 0
        assert converter.z_today.unit == cu.redshift

    def test_a_today(self, converter):
        """Test ``DistanceMeasureConverter.a_today``."""
        assert converter.a_today == 1
        assert converter.a_today.unit == u.dimensionless_unscaled

    def test_rho_today(self, converter):
        """Test ``DistanceMeasureConverter.rho_today``."""
        assert converter.rho_today == converter.rho_of_z(0)
        assert converter.rho_today.unit == u.dimensionless_unscaled

    def test_z_of_a(self, converter):
        """Test ``DistanceMeasureConverter.z_of_a``."""
        # Edge cases
        assert converter.z_of_a(1) == 0
        assert converter.z_of_a(0) == inf

        # Consistency
        assert converter.z_of_a(converter.a_today) == converter.z_today
        assert (
            converter.z_of_a(converter.a_matter_radiation_equality)
            == converter.z_matter_radiation_equality
        )
        assert converter.z_of_a(converter.a_begin) == converter.z_begin

    def test_a_of_z(self, converter):
        """Test ``DistanceMeasureConverter.a_of_z``."""
        # Edge cases
        assert converter.a_of_z(0) == 1
        assert converter.a_of_z(inf) == 0

        # Consistency
        assert converter.a_of_z(converter.z_today) == converter.a_today
        assert (
            converter.a_of_z(converter.z_matter_radiation_equality)
            == converter.a_matter_radiation_equality
        )
        assert converter.a_of_z(converter.z_begin) == converter.a_begin

    def test_a_of_rho(self, converter):
        """Test ``DistanceMeasureConverter.a_of_rho``."""
        # Edge cases
        assert converter.a_of_rho(1 / sqrt(2)).value == approx(0)

        # Consistency
        assert converter.a_of_rho(converter.rho_today).value == approx(
            converter.a_today.value
        )
        assert (
            converter.a_of_rho(converter.rho_matter_radiation_equality)
            == converter.a_matter_radiation_equality
        )
        assert converter.a_of_rho(converter.rho_begin).value == approx(
            converter.a_begin.value
        )

    def test_rho_of_a(self, converter):
        """Test ``DistanceMeasureConverter.rho_of_a``."""
        # Edge cases
        assert converter.rho_of_a(1).value == approx(41.16336547)
        assert converter.rho_of_a(0).value == approx(1 / sqrt(2))

        # Consistency
        assert converter.rho_of_a(converter.a_today) == converter.rho_today
        assert (
            converter.rho_of_a(converter.a_matter_radiation_equality)
            == converter.rho_matter_radiation_equality
        )
        assert converter.rho_of_a(converter.a_begin).value == approx(
            converter.rho_begin.value
        )

    def test_rho_of_z(self, converter):
        """Test ``DistanceMeasureConverter.rho_of_z``."""
        # Edge cases
        assert converter.rho_of_z(0).value == approx(41.16336547)
        assert converter.rho_of_z(inf).value == approx(1 / sqrt(2))

        # Consistency
        assert converter.rho_of_z(converter.z_today) == converter.rho_today
        assert (
            converter.rho_of_z(converter.z_matter_radiation_equality)
            == converter.rho_matter_radiation_equality
        )
        assert converter.rho_of_z(converter.z_begin).value == approx(
            converter.rho_begin.value
        )

    def test_z_of_rho(self, converter):
        """Test ``DistanceMeasureConverter.z_of_rho``."""
        # Edge cases
        assert converter.z_of_rho(1 / sqrt(2)) == inf
        assert converter.z_of_rho(41.16336547) == 0

        # Consistency
        assert converter.z_of_rho(converter.rho_today).value == approx(
            converter.z_today.value
        )
        assert (
            converter.z_of_rho(converter.rho_matter_radiation_equality)
            == converter.z_matter_radiation_equality
        )
        assert converter.z_of_rho(converter.rho_begin) == converter.z_begin

    # ========================================================================

    def test_init(self, converter_cls, cosmo, cosmo_proxy):
        """Test ``DistanceMeasureConverter``."""
        inst = converter_cls(cosmo)
        assert inst.cosmo is cosmo

        inst = converter_cls(cosmo_proxy)
        assert inst.cosmo is cosmo_proxy
