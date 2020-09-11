# -*- coding: utf-8 -*-

"""Unsorted Utilities."""

__all__ = [
    "flatten_dict",
    "z_matter_radiation_equality",
    "zeta_of_z"
]


##############################################################################
# IMPORTS

# BUILT-IN

import typing as T


# THIRD PARTY

from astropy.cosmology import default_cosmology
import numpy as np


# PROJECT-SPECIFIC


##############################################################################
# PARAMETERS


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


# -------------------------------------------------------------------


def z_matter_radiation_equality(
    cosmo,
    zmin: float = 1e3,
    zmax: float = 1e4,
    full_output: bool = False,
    **rootkw,
):
    """Calculate matter-radiation equality for a given cosmology.

    Parameters
    ----------
    cosmo : Cosmology
        Must have methods ``Om(z)`` and ``Ogamma(z)``
    zmin, zmax : float
        The min/max z in which to search for z-equality
    full_output : bool
        Whether to return the full output of the scipy rootfinder,
        or just z-equality
    rootkw
        kwargs into scipy minimizer. See `~scipy.optimize.brentq` for details

    Returns
    -------
    z_eq : float
        If `full_output` is False
    Any
        Full results of root finder if `full_outpyt` is True

    """
    # import rootfinder
    from scipy.optimize import brentq

    # residual function
    def f(z):
        return cosmo.Om(z) - cosmo.Ogamma(z)

    return brentq(f, zmin, zmax, full_output=full_output, **rootkw)


# /def


# -------------------------------------------------------------------


def zeta_of_z(
    z: T.Union[float, np.ndarray], zeq: T.Union[float, np.ndarray, None] = None
) -> T.Union[float, np.ndarray]:
    r"""Reduced redshift, scaled by matter-radiation equality.

    :math:`\zeta = \frac{1+z}{1+z_{\mathrm{eq}}}`

    .. |ndarray| replace:: `~numpy.ndarray`

    Parameters
    ----------
    z : float or |ndarray|
        The redshift
    zeq : float or |ndarray| or None
        The redshift at matter-radiation equality.
        If None (default), calculates from the
        `~astropy.cosmology.default_cosmolgy`.

    Returns
    -------
    zeta : float or |ndarray|
        ndarray if either `z` or `zeq` is ndarray, else float.

    """
    if zeq is None:
        zeq: float = z_matter_radiation_equality(default_cosmology.get())

    return (1.0 + z) / (1.0 + zeq)


# /def


# -------------------------------------------------------------------

##############################################################################
# END
