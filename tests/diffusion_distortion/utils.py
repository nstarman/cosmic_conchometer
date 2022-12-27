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
        z = np.arange(0, 1000, 100)
        return {"z": z, "exp(-kappa)": 1 / (1 + z), "g [Mpc^-1]": z / (1 + z)}
