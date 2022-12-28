# LOCAL
from cosmic_conchometer.utils.classy_utils import (
    CLASSConfigParser,
    _flatten_dict,
    read_params_from_ini,
)


def test__flatten_dict():
    """Test :func:`~cosmic_conchometer.utils.classy_utils._flatten_dict`."""
    # Simple case
    d = {"a": 1, "b": 2, "c": 3}

    assert _flatten_dict(d) == d

    # Nested case
    d = {"a": 1, "_": {"c": 2, "d": 3}}

    assert _flatten_dict(d) == {"a": 1, "c": 2, "d": 3}

    # Nested case
    d = {"a": 1, "_": {"c": 2, "d": 3}, "__": {"f": 4, "g": 5}}

    assert _flatten_dict(d) == {"a": 1, "c": 2, "d": 3, "f": 4, "g": 5}


def test_CLASSConfigParser():
    """Test :class:`~cosmic_conchometer.utils.classy_utils.CLASSConfigParser`."""
    # Simple case
    parser = CLASSConfigParser()
    parser.read_string(
        """
        [background parameters]
        h =0.674
        T_cmb = 2.7255
        """
    )

    assert parser["background parameters"]["h"] == "0.674"
    assert parser["background parameters"]["T_cmb"] == "2.7255"


def test_read_params_from_ini(tmp_path):
    """Test :func:`~cosmic_conchometer.utils.classy_utils.read_params_from_ini`."""
    p = tmp_path / "config.ini"
    p.write_text(
        """
        [background parameters]
        h =0.674
        T_cmb = 2.7255
        """
    )

    params = read_params_from_ini(p)

    assert params["h"] == "0.674"
    assert params["T_cmb"] == "2.7255"
