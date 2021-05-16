# -*- coding: utf-8 -*-

"""Initiation Tests for `~cosmic_conchometer`."""

__all__ = [
    "test_has_version",
    "test_setup_package_flags",
]

##############################################################################
# IMPORTS

# BUILT-IN
import pathlib

##############################################################################
# TESTS
##############################################################################


def test_has_version():
    """The most basic test."""
    # PROJECT-SPECIFIC
    import cosmic_conchometer

    assert hasattr(cosmic_conchometer, "__version__"), "No version!"


# /def

##############################################################################
# END
