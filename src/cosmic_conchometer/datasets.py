"""Load sample data."""

# STDLIB
from typing import Any

# THIRD-PARTY
import pooch
from importlib_metadata import version as get_version

# LOCAL
from cosmic_conchometer.utils import classy_utils

__all__ = ["fetch_planck18_parameters"]

cc_data = pooch.create(
    # Use the default cache folder for the operating system
    path=pooch.os_cache("cosmic_conchometer"),
    # The remote data is on Github
    base_url="https://github.com/nstarman/cosmic_conchometer/raw/{version}/data/",
    version=get_version("cosmic_conchometer"),
    # If this is a development version, get the data from the "main" branch
    version_dev="main",
    registry={
        "planck18_parameters.ini": "sha256:80544a579b2eeb898e4b1f4a3de4962e5b2d3ee92073562bd4268304eb97d749",  # noqa: E501
    },
)


def fetch_planck18_parameters() -> dict[str, Any]:
    """Load the Planck'18 parameters."""
    # The file will be downloaded automatically the first time this is run
    # returns the file path to the downloaded file. Afterwards, Pooch finds
    # it in the local cache and doesn't repeat the download.
    fname = cc_data.fetch("planck18_parameters.ini")
    # The "fetch" method returns the full path to the downloaded data file.
    # All we need to do now is load it with our standard Python tools.
    data = classy_utils.read_params_from_ini(fname)
    return data
