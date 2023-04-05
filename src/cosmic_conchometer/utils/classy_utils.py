"""Utilities for :mod:`classy`."""

from __future__ import annotations

import configparser
from typing import TYPE_CHECKING, Any

__all__ = ["read_params_from_ini"]

if TYPE_CHECKING:
    from os import PathLike

##############################################################################
# CODE
##############################################################################


def _flatten_dict(d: dict[str, Any | dict[str, Any]], /) -> dict[str, Any]:
    """Recursively flatten nested dictionary.

    Parameters
    ----------
    d : dict[str, Any | dict[str, Any]], position-only
        Dictionary to flatten.

    Returns
    -------
    out : dict[str, Any]
        Flattened dictionary.

    Examples
    --------
    >>> _flatten_dict({"a": 1, "b": 2, "c": 3})
    {'a': 1, 'b': 2, 'c': 3}

    >>> _flatten_dict({"a": 1, "_": {"c": 2, "d": 3}})
    {'a': 1, 'c': 2, 'd': 3}
    """
    out: dict[str, Any] = {}
    for key, val in d.items():
        out.update(_flatten_dict(val) if isinstance(val, dict) else {key: val})
    return out


class CLASSConfigParser(configparser.ConfigParser):
    """Parser for CLASS config files.

    This is a subclass of :class:`configparser.ConfigParser` that overrides
    :meth:`optionxform` to aalways return the string form of the input. This
    is necessary because CLASS config files should stay as strings.

    Examples
    --------
    An example file::

        [background parameters]

        # Hubble parameter:
        h =0.674
        # Photon density:
        T_cmb = 2.7255
        # Baryon density:
        omega_b = 0.0224
        # Ultra-relativistic species / massless neutrino density:
        N_ur = 3.046
        # Density of cdm (cold dark matter):
        omega_cdm = 0.12
        # Curvature:
        Omega_k = 0.
        # Scale factor today 'a_today'
        a_today = 1.

        ...
    """

    def optionxform(self, optionxform: str) -> str:
        """Return string as-is."""
        return str(optionxform)


def read_params_from_ini(filename: str | bytes | PathLike[str], /) -> dict[str, Any]:
    """Read parameters from INI file.

    Parameters
    ----------
    filename : str | bytes | PathLike[str], position-only
        Path to INI file.

    Returns
    -------
    params : dict[str, Any]
        Dictionary of parameters. Nested dict are flattened.
    """
    config = CLASSConfigParser()
    config.read(str(filename))
    return _flatten_dict({k: dict(config[k]) for k in config.sections()})
