"""Tests for :mod:`~cosmic_conchometer.ps`."""

# THIRD-PARTY
import pytest

# LOCAL
from cosmic_conchometer.params import planck18
from cosmic_conchometer.tf import baumann_transfer_function

##############################################################################
# TESTS
##############################################################################


@pytest.mark.parametrize(("kmag", "z_last_scatter", "expect"), [(0, 1100, -0.2)])
def test_baumann_transfer_function(kmag, z_last_scatter, expect):
    """Test :func:`~cosmic_conchometer.tf.baumann_transfer_function`."""
    assert (
        baumann_transfer_function(planck18, kmag, z_last_scatter=z_last_scatter)
        == expect
    )
