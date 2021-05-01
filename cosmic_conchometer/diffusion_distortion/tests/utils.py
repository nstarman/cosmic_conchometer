# -*- coding: utf-8 -*-

"""Utilities for testing :mod:`~cosmic_conchometer.diffusion_distortion`."""

__all__ = [
    "CLASS",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import numpy as np

##############################################################################
# CODE
##############################################################################


class CLASS:
    """Placeholder CLASS cosmology."""

    def get_thermodynamics(self):
        """Make fake thermodynamics data."""
        z = np.arange(0, 1000, 100)
        return {"z": z, "exp(-kappa)": 1 / (1 + z), "g [Mpc^-1]": z / (1 + z)}

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
