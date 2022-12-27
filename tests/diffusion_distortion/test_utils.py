"""Tests for :mod:`~cosmic_conchometer.ps`."""

# THIRD-PARTY
import pytest

# LOCAL
from cosmic_conchometer.diffusion_distortion.utils import rho2_of_rho1

##############################################################################
# TESTS
##############################################################################


@pytest.mark.skip(reason="Not implemented")
def test_rho2_of_rho1(kmag, z_last_scatter, expect):
    """Test :func:`~cosmic_conchometer.tf.baumann_transfer_function`."""
    assert rho2_of_rho1(0, 0, 0, maxrho_domain=0) == expect
