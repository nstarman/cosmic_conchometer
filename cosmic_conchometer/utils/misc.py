# -*- coding: utf-8 -*-

"""Unsorted Utilities."""

__all__ = [
    "flatten_dict",
]


##############################################################################
# IMPORTS

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
