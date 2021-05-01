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


def test_setup_package_flags():
    """Look for flags used in setting up the package."""
    # PROJECT-SPECIFIC
    from cosmic_conchometer import setup_package as flags

    # tqdm
    try:
        # THIRD PARTY
        from tqdm import tqdm  # noqa: F401
    except ImportError:
        assert flags.HAS_TQDM is False
    else:
        assert flags.HAS_TQDM is True

    # meta-test that this test is capturing all the flags
    IS_TESTED = ["TQDM"]
    assert all([f in IS_TESTED for f in flags.__all__ if f.upper() == f])


# /def


##############################################################################
# END
