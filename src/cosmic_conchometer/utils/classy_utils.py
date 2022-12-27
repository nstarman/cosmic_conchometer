"""Utilities for :mod:`classy`."""

from __future__ import annotations

# STDLIB
import configparser
import os
from typing import Any

__all__ = ["read_params_from_ini"]

##############################################################################
# CODE
##############################################################################


def _flatten_dict(d: dict[str, Any | dict[str, Any]]) -> dict[str, Any]:
    """Recursively flatten nested dictionary."""
    out: dict[str, Any] = {}

    for key, val in d.items():
        if isinstance(val, dict):
            out.update(_flatten_dict(val))
        else:
            out[key] = val

    return out


class CLASSConfigParser(configparser.ConfigParser):
    """Parser for CLASS config files."""

    def optionxform(self, optionxform: str) -> str:
        """Return string as-is."""
        return str(optionxform)


def read_params_from_ini(filename: str | bytes | os.PathLike[str]) -> dict[str, Any]:
    """Read parameters from INI file."""
    config = CLASSConfigParser()
    config.read(str(filename))

    params = _flatten_dict({k: dict(config[k]) for k in config.sections()})
    return params
