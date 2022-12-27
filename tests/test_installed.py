"""Initiation Tests for `~cosmic_conchometer`."""


def test_has_version():
    """The most basic test."""
    # LOCAL
    import cosmic_conchometer

    assert hasattr(cosmic_conchometer, "__version__"), "No version!"
