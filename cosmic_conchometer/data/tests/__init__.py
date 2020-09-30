# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Tests for :mod:`~cosmic_conchometer.data`."""

__all__ = [
    # modules
    # instance
    "test",
]


##############################################################################
# IMPORTS

# BUILT-IN
from pathlib import Path

# THIRD PARTY
from astropy.tests.runner import TestRunner

##############################################################################
# TESTS
##############################################################################

test = TestRunner.make_test_runner_in(Path(__file__).parent.parent)

##############################################################################
# END
