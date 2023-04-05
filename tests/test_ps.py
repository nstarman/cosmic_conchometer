"""Tests for :mod:`~cosmic_conchometer.ps`."""


from math import inf

import pytest
from cosmic_conchometer.params import planck18
from cosmic_conchometer.ps import power_spectrum

##############################################################################
# TESTS
##############################################################################


@pytest.mark.parametrize(("kmag", "pivot_scale", "expect"), [(0, 1, inf), (inf, 1, 0)])
def test_power_spectrum(kmag, pivot_scale, expect):
    """Test :func:`~cosmic_conchometer.ps.power_spectrum`."""
    assert power_spectrum(planck18, kmag, pivot_scale=pivot_scale) == expect
