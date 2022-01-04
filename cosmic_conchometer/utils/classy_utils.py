# -*- coding: utf-8 -*-

"""Utilities for :mod:`classy`."""

__all__ = ["read_params_from_ini"]

##############################################################################
# IMPORTS

# STDLIB
import configparser

##############################################################################
# CODE
##############################################################################


def _flatten_dict(d: dict) -> dict:
    """Recursively flatten nested dictionary."""
    out: dict = {}
    for key, val in d.items():
        if type(val) == dict:
            out.update(_flatten_dict(val))
        else:
            out[key] = val

    return out


class CLASSConfigParser(configparser.ConfigParser):
    def optionxform(self, optionxform: str) -> str:
        """Return string as-is."""
        return str(optionxform)


def read_params_from_ini(filename: str) -> dict:
    """Read parameters from INI file.

    Parameters
    ----------
    filename : str

    Returns
    -------
    dict
    """
    config = CLASSConfigParser()
    config.read("input/parameters.ini")

    params = _flatten_dict({k: dict(config[k]) for k in config.sections()})
    return params


##############################################################################
# END
