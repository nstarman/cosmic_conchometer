"""Utilities for testing :mod:`~cosmic_conchometer.diffusion_distortion`."""

##############################################################################
# IMPORTS

# STDLIB
from typing import Any

# THIRD-PARTY
import numpy as np

##############################################################################
# CODE
##############################################################################


class CLASS:
    """Placeholder CLASS cosmology."""

    def get_thermodynamics(self) -> dict[str, Any]:
        """Make fake thermodynamics data."""
        z = np.unique(np.geomspace(1, 3000, 10_000))
        return {
            "z": z,
            "g [Mpc^-1]": 50 * np.exp(-((z - 1089) ** 2) / 200),
        }
