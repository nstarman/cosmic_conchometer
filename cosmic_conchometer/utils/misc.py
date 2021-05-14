# -*- coding: utf-8 -*-

"""Unsorted Utilities."""

__all__ = [
    "flatten_dict",
]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T

# THIRD PARTY
import numpy as np
from astropy.cosmology import default_cosmology
from astropy.cosmology.core import Cosmology

##############################################################################
# CODE
##############################################################################


def flatten_dict(d: dict) -> dict:
    """Recursively flatten nested dictionary."""
    out: dict = {}
    for key, val in d.items():
        if type(val) == dict:
            out.update(flatten_dict(val))
        else:
            out[key] = val

    return out


# /def

##############################################################################
# END
