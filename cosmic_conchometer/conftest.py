# -*- coding: utf-8 -*-
"""Configure Test Suite.

This file is used to configure the behavior of pytest when using the Astropy
test infrastructure. It needs to live inside the package in order for it to
get picked up when running the tests inside an interpreter using
packagename.test

"""

# BUILT-IN
import os
import typing as T

# THIRD PARTY
import pytest
from astropy.version import version as astropy_version

# For Astropy 3.0 and later, we can use the standalone pytest plugin
if astropy_version < "3.0":
    # THIRD PARTY
    from astropy.tests.pytest_plugins import *  # noqa: F401, F403

    del pytest_report_header  # type: ignore
    ASTROPY_HEADER = True
else:
    try:
        # THIRD PARTY
        from pytest_astropy_header.display import (
            PYTEST_HEADER_MODULES,
            TESTED_VERSIONS,
        )

        ASTROPY_HEADER = True
    except ImportError:
        ASTROPY_HEADER = False


def pytest_configure(config: T.Any) -> None:
    """Configure Pytest with Astropy.

    Parameters
    ----------
    config : pytest configuration

    """
    if ASTROPY_HEADER:

        config.option.astropy_header = True

        # Customize the following lines to add/remove entries from the list of
        # packages for which version numbers are displayed when running the tests.
        PYTEST_HEADER_MODULES.pop("Pandas", None)
        PYTEST_HEADER_MODULES["scikit-image"] = "skimage"

        # PROJECT-SPECIFIC
        from . import __version__

        packagename = os.path.basename(os.path.dirname(__file__))
        TESTED_VERSIONS[packagename] = __version__


# /def


# ------------------------------------------------------


@pytest.fixture(autouse=True)
def add_astropy(doctest_namespace: T.Any) -> None:
    """Add Imports to Pytest.

    Parameters
    ----------
    doctest_namespace : namespace

    """
    # import
    # THIRD PARTY
    import astropy.coordinates
    import astropy.units

    # add to namespace
    doctest_namespace["u"] = astropy.units
    doctest_namespace["coord"] = astropy.coordinates


# def
