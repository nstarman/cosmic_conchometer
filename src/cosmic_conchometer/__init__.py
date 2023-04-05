# see LICENSE.rst

"""Cosmic Conchometer -- Measurements in Last-Scattering Shells."""

from __future__ import annotations

import contextlib
from importlib.metadata import version as _get_version

__all__: list[str] = []

__author__ = ["Nathaniel Starkman", "Glenn Starkman", "Arthur Kosowsky"]
__copyright__ = "Copyright 2020, "
__maintainer__ = "Nathaniel Starkman"
__email__ = "n[dot]starkman[at]mail.utoronto.ca"
__version__ = _get_version("cosmic_conchometer")

# Clean all private names
for _n in list(globals()):
    if _n.startswith("_") and not _n.startswith("__"):
        with contextlib.suppress(KeyError):
            del globals()[_n]
