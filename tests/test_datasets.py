"""Tests for :mod:`~cosmic_conchometer.datasets`."""

# THIRD-PARTY
import pooch
import pytest
from classy import Class as CLASS

# LOCAL
from cosmic_conchometer.datasets import cc_data, fetch_planck18_parameters

##############################################################################
# TESTS
##############################################################################


def test_pooch():
    """Test Pooch object."""
    assert isinstance(cc_data, pooch.Pooch)


@pytest.mark.skip(reason="TODO!")
def test_fetch_planck18_parameters():
    """Test :func:`~cosmic_conchometer.datasets.fetch_planck18_parameters`."""
    params = fetch_planck18_parameters()

    class_cosmo = CLASS()
    class_cosmo.set(params)

    # Run the whole code.
    class_cosmo.compute()

    # TODO! more tests
