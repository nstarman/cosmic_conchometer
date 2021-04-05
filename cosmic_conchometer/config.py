# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Configuration."""


##############################################################################
# IMPORTS

# THIRD PARTY
from astropy import config as _config

__all__ = ["conf"]

#############################################################################
# CONFIGURATIONS


class Conf(_config.ConfigNamespace):
    """Configuration parameters for :mod:`~starkman_thesis`."""

    default_cosmo = _config.ConfigItem(
        "Planck18",
        description="Default Cosmology.",
        cfgtype="string",
    )


conf = Conf()
# /class

#############################################################################
# END
