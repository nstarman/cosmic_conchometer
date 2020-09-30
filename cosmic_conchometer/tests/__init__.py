# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Tests for :mod:`~cosmic_conchometer`."""

__all__ = [
    # modules
    "init_tests",
    # instance
    "test",
]


##############################################################################
# IMPORTS

# BUILT-IN
from pathlib import Path

# THIRD PARTY
from astropy.tests.runner import TestRunner

# PROJECT-SPECIFIC
from . import test_init as init_tests

##############################################################################
# TESTS
##############################################################################

test = TestRunner.make_test_runner_in(Path(__file__).parent.parent)

##############################################################################
# END
