"""Tests for :mod:`~cosmic_conchometer.ps`."""


from math import sqrt

import numpy as np
import pytest
from cosmic_conchometer.temperature_diffusion_spectra_distortion.utils import (
    rho2_of_rho1,
)


##############################################################################
# TESTS
##############################################################################


@pytest.mark.parametrize(
    ("rho", "spll", "sprp", "maxrho_domain", "expect"),
    [
        # Analytic results
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0, 0.0, -1.0),
        (1.0, 0.0, 1.0, 0.0, 1 - sqrt(2)),
        (1.0, 1.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, 1.0, 1.0, 0.0),
        # Cycling through broadcasting
        (np.array([0, 0]), 0.0, 0.0, 0.0, np.array([0, 0])),
        (0.0, np.array([0, 0]), 0.0, 0.0, np.array([0, 0])),
        (0.0, 0.0, np.array([0, 0]), 0.0, np.array([0, 0])),
        (0.0, 0.0, 0.0, np.array([0, 0]), np.array([0, 0])),
    ],
)
def test_rho2_of_rho1(rho, spll, sprp, maxrho_domain, expect):
    """Test :func:`~cosmic_conchometer.tf.baumann_transfer_function`."""
    assert rho2_of_rho1(rho, spll, sprp, maxrho_domain=maxrho_domain) == pytest.approx(
        expect
    )
