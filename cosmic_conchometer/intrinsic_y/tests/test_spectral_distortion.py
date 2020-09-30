# -*- coding: utf-8 -*-

"""Initiation Tests for `~cosmic_conchometer`."""

__all__ = [
    "test__IUSType",
]


##############################################################################
# IMPORTS

import collections.abc as cabc
import typing as T

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

from cosmic_conchometer import intrinsic_y

from .test_core import Test_IntrinsicDistortionBase

##############################################################################
# TESTS
##############################################################################


def test__IUSType():
    """Test `~cosmic_conchometer.intrinsic_y._spectral_distortion._IUSType`."""
    _IUSType = intrinsic_y._spectral_distortion._IUSType

    assert T.get_origin(_IUSType) == cabc.Callable

    args = T.get_args(_IUSType)

    # input
    assert isinstance(args[0], list)
    assert T.get_origin(args[0][0]) == T.Union
    assert T.get_args(args[0][0]) == (float, np.ndarray)
    # output
    assert args[1] == np.ndarray


# /def


# -------------------------------------------------------------------


class Test_SpectralDistortion(Test_IntrinsicDistortionBase):
    """Test `~cosmic_conchometer.intrinsic_y.SpectralDistortion`."""

    _cls = intrinsic_y.SpectralDistortion

    def test_instantiation(self):
        """Test class instantiation options, including exceptions."""
        # Most basic instantiation.
        self._cls(self.cosmo, self.class_cosmo)

        # defer tests covered by tests of IntrinsicDistortionBase.

    # /def

    def test_attributes(self):
        """Test class has expected attributes."""
        # make class
        idb = self._cls(self.cosmo, self.class_cosmo)

        # test has attributes
        # defer tests of IntrinsicDistortionBase variables for that test suite.
        for attr in (
            "PgamBarCL",
            "GgamBarCL",
            "PgamBarCL0",
            "angular_summand",
        ):
            hasattr(idb, attr)

    # /def

    def test_prefactor(self):
        """Test `~cosmic_conchometer.intrinsic_y.SpectralDistortion.prefactor`."""
        # Should be NaN when frequency is 0
        assert np.isnan(self.instance.prefactor(0 * u.GHz, 1 / u.Mpc))

        # Frequency of s.t. reduced_energy is 1.
        unity_freq = 1.0 * (const.k_B * self.instance.Tcmb0) / const.h << u.Hz
        reduced_energy = (
            unity_freq * const.h / (const.k_B * self.instance.Tcmb0)
        ) << u.one
        expected = (
            reduced_energy
            / np.expm1(-reduced_energy)
            * self.instance.lambda0 ** 2
            / (16 * np.pi * self.instance.PgamBarCL0)
        )
        assert self.instance.prefactor(unity_freq, 1 / u.Mpc) == expected

        # test when AkFunc is not unity
        from cosmic_conchometer import default_Ak

        with default_Ak.set(lambda x: 0.5):
            self.instance.AkFunc = default_Ak.get()
            assert (
                self.instance.prefactor(unity_freq, 1 / u.Mpc)
                == expected / 2.0
            )
        # reset
        self.instance.AkFunc = default_Ak.get()

    # /def

    @pytest.mark.skip()
    def test__angular_summand(self):
        """Test `~cosmic_conchometer.intrinsic_y.SpectralDistortion._angular_summand`."""
        pass

    # /def

    @pytest.mark.skip()
    def test_angular_sum(self):
        """Test `~cosmic_conchometer.intrinsic_y.SpectralDistortion.angular_sum`."""
        pass

    # /def

    @pytest.mark.skip()
    def test__emission_integrand(self):
        """Test `~cosmic_conchometer.intrinsic_y.SpectralDistortion._emission_integrand`."""
        pass

    # /def

    @pytest.mark.skip()
    def test_emission_integral(self):
        """Test `~cosmic_conchometer.intrinsic_y.SpectralDistortion.emission_integral`."""
        pass

    # /def

    @pytest.mark.skip()
    def test__scatter_integrand(self):
        """Test `~cosmic_conchometer.intrinsic_y.SpectralDistortion._scatter_integrand`."""
        pass

    # /def

    @pytest.mark.skip()
    def test_scatter_integral(self):
        """Test `~cosmic_conchometer.intrinsic_y.SpectralDistortion.scatter_integral`."""
        pass

    # /def

    @pytest.mark.skip()
    def test_compute(self):
        """Test `~cosmic_conchometer.intrinsic_y.SpectralDistortion.compute`.

        Compute is same as ``__call__``.

        """
        pass

    # /def

    @pytest.mark.skip()
    def test_plot_PgamBarCL(self):
        """Test `~cosmic_conchometer.intrinsic_y.SpectralDistortion.plot_PgamBarCL`.

        .. todo::

            Move this to a special plots test file.

        """
        pass

    # /def

    @pytest.mark.skip()
    def test_plot_GgamBarCL(self):
        """Test `~cosmic_conchometer.intrinsic_y.SpectralDistortion.plot_GgamBarCL`.

        .. todo::

            Move this to a special plots test file.

        """
        pass

    # /def


# /class


##############################################################################
# END
