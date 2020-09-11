# -*- coding: utf-8 -*-
# see LICENSE.rst

"""cosmic_conchometer."""

__author__ = ["Nathaniel Starkman", "Glenn Starkman", "Arthur Kosowsky"]
__copyright__ = "Copyright 2020, "
__maintainer__ = "Nathaniel Starkman"
__email__ = "n[dot]starkman[at]mail.utoronto.ca"

__all__ = [
    # modules
    "utils",
    # functions
]


##############################################################################
# IMPORTS

# Packages may add whatever they like to this file, but
# should keep this content at the top.
from ._astropy_init import *  # noqa  # isort:skip

from . import core, utils
from .common import *
from .core import *

# __all__
__all__ += common.__all__
__all__ += core.__all__


##############################################################################
# END
