# -*- coding: utf-8 -*-

"""Initiation Tests for `~cosmic_conchometer`."""

__all__ = [
    "test_has_version",
    "test_setup_package_flags",
]

##############################################################################
# IMPORTS

# STDLIB
import pathlib

# PROJECT-SPECIFIC
from cosmic_conchometer import setup_package

##############################################################################
# TESTS
##############################################################################


def test_setup_package_flags():
    """Look for flags used in setting up the package."""
    # ------------
    # data directory

    assert isinstance(setup_package.DATA_DIR, pathlib.Path)

    # ------------
    # tqdm
    try:
        # THIRD PARTY
        from tqdm import tqdm  # noqa: F401
    except ImportError:
        assert setup_package.HAS_TQDM is False
    else:
        assert setup_package.HAS_TQDM is True

    # ------------

    # meta-test that this test is capturing all the flags
    IS_TESTED = ["HAS_TQDM", "DATA_DIR"]
    assert all([f in IS_TESTED for f in setup_package.__all__ if f.upper() == f])


# /def

# -------------------------------------------------------------------


class TestNoOpBar:

    # ============================================
    # Method Tests

    def test___init__(self):
        """Test method ``__init__``."""
        # basic
        pbar = setup_package._NoOpPBar()

        # args and kwargs
        pbar = setup_package._NoOpPBar(1, 2, 3, a=1, b=2, c=3)

    def test___enter__and__exit__(self):
        """Test methods ``__enter__`` and ``__exit__``."""
        pbar = setup_package._NoOpPBar()

        with pbar(1, 2, 3, a=1, b=2, c=3):
            pass

    def test_update(self):
        """Test method ``update``."""
        pbar = setup_package._NoOpPBar()
        pbar.update(1)

    # ============================================
    # Usage Test

    def test_usage(self):
        """Test usage of progress bar."""
        pbar = setup_package._NoOpPBar(1, 2, 3, a=1, b=2, c=3)

        with pbar(1, 2, 3, a=1, b=2, c=3):
            pbar.update(1)


##############################################################################
# END
