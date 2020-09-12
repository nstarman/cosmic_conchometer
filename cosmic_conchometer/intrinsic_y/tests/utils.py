# -*- coding: utf-8 -*-

"""Utilities for Testing :mod:`~cosmic_conchometer.intrinsic_y`."""

__all__ = [
    "CLASS",
]


##############################################################################
# IMPORTS

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
