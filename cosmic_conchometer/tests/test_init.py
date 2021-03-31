# -*- coding: utf-8 -*-

"""Initiation Tests for `~cosmic_conchometer`."""

__all__ = [
    "test_has_version",
]


##############################################################################
# IMPORTS

# BUILT-IN

# THIRD PARTY

# PROJECT-SPECIFIC


##############################################################################
# PARAMETERS


##############################################################################
# TESTS
##############################################################################


def test_has_version():
    """The most basic test."""
    # PROJECT-SPECIFIC
    import cosmic_conchometer

    assert hasattr(cosmic_conchometer, "__version__"), "No version!"


# /def


# -------------------------------------------------------------------


##############################################################################
# END
