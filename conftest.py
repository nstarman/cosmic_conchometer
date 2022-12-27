"""Configure Test Suite.

This file is used to configure the behavior of pytest when using the Astropy
test infrastructure. It needs to live inside the package in order for it to
get picked up when running the tests inside an interpreter using
packagename.test

"""

from __future__ import annotations

# STDLIB
import os
from importlib.metadata import version
from typing import Any

# THIRD-PARTY
import pytest
from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS


def pytest_configure(config: pytest.Config) -> None:
    """Configure Pytest with Astropy."""
    config.option.astropy_header = True

    # Customize the following lines to add/remove entries from the list of
    # packages for which version numbers are displayed when running the
    # tests.
    PYTEST_HEADER_MODULES.pop("Pandas", None)

    packagename = os.path.basename(os.path.dirname(__file__))
    TESTED_VERSIONS[packagename] = version("cosmic_conchometer")


@pytest.fixture(autouse=True)  # type: ignore[misc]
def add_numpy(doctest_namespace: dict[str, Any]) -> None:
    """Add NumPy to Pytest."""
    # THIRD-PARTY
    import numpy as np

    # add to namespace
    doctest_namespace["np"] = np


@pytest.fixture(autouse=True)  # type: ignore[misc]
def add_astropy(doctest_namespace: dict[str, Any]) -> None:
    """Add imports to Pytest."""
    # THIRD-PARTY
    import astropy.coordinates
    import astropy.units

    # add to namespace
    doctest_namespace["u"] = astropy.units
    doctest_namespace["coord"] = astropy.coordinates
