# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Cosmic Conchometer -- Measurements in Last-Scattering Shells."""

__author__ = ["Nathaniel Starkman", "Glenn Starkman", "Arthur Kosowsky"]
__copyright__ = "Copyright 2020, "
__maintainer__ = "Nathaniel Starkman"
__email__ = "n[dot]starkman[at]mail.utoronto.ca"

__all__ = [
    # modules
    "boltzmann",
    "common",
    "data",
    "diffusion_distortion",
    "initial_conditions",
    "utils",
    # objects
    "conf",
    # etc.
    "DATA_DIR",
]

##############################################################################
# IMPORTS

# ---------------------------------------------------------
# Packages may add whatever they like to this file, but
# should keep this content at the top.
from ._astropy_init import *  # noqa: F401, F403  # isort:skip

# ---------------------------------------------------------

# PROJECT-SPECIFIC
from . import boltzmann, common, data, diffusion_distortion, initial_conditions, utils
from .common import *  # noqa: F401, F403
from .config import conf
from .setup_package import DATA_DIR

# ALL
__all__ += common.__all__


##############################################################################
# END
