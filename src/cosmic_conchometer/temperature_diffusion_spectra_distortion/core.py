"""Intrinsic Distortion Core Functions."""

from __future__ import annotations

import itertools
from abc import ABCMeta
from collections.abc import Callable, Mapping
from dataclasses import InitVar, dataclass
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast
from weakref import proxy

import astropy.cosmology.units as cu
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FLRW  # noqa: TCH002
from astropy.utils.metadata import MetaData
from classy import Class as Classy  # noqa: TCH002
from scipy.interpolate import InterpolatedUnivariateSpline as IUSpline

from cosmic_conchometer.temperature_diffusion_spectra_distortion.prob_2ls import (
    ComputePspllSprp,
)
from cosmic_conchometer.utils.distances import DistanceMeasureConverter

if TYPE_CHECKING:
    from astropy.units import Quantity
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike, NDArray


__all__: list[str] = []


##############################################################################
# PARAMETERS

IUSType = Callable[["ArrayLike"], "NDArray[np.floating[Any]]"]


##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class SpectralDistortion(metaclass=ABCMeta):
    r"""Spectral Distortion.

    Parameters
    ----------
    cosmo : `~astropy.cosmology.core.Cosmology`
        Astroy cosmology instance.
    class_cosmo : :class:`classy.Class`
        CLASS cosmoloy instance that corresponds to ``cosmo``.
    z_domain : tuple[float, float]
        Redshift domain to consider. The first element is the maximum redshift, the
        second is the minimum redshift.
    """

    cosmo: FLRW
    class_cosmo: Classy
    z_bounds: InitVar[tuple[float, float]] = (5500, 100)

    meta = MetaData()

    def __post_init__(self, z_bounds: tuple[float, float]) -> None:
        self.distance_converter: DistanceMeasureConverter
        object.__setattr__(
            self, "distance_converter", DistanceMeasureConverter(proxy(self.cosmo))
        )

        # --- thermodynamic quantities --- #

        th = self.class_cosmo.get_thermodynamics()

        # time-order everything & cut to be in z_domain (inplace)
        _in_bounds = (min(z_bounds) <= th["z"]) & (th["z"] <= max(z_bounds))
        for k, v in th.items():
            th[k] = v[_in_bounds][::-1]
            th[k].flags.writeable = False

        # Rename and assign units
        it: tuple[tuple[str, str, u.UnitBase], ...] = (
            ("z", "z", cu.redshift),
            ("g [Mpc^-1]", "g", u.Unit("1 / Mpc")),
        )
        for name, newname, unit in it:
            th[newname] = th.pop(name) * unit

        # Add derived values
        th["rho"] = self.distance_converter.rho_of_z(th["z"])

        # Ensure everything immutable again, b/c added stuff
        for k in th:
            th[k].flags.writeable = False

        self._class_thermo: Mapping[str, NDArray[np.floating[Any]]]
        object.__setattr__(self, "_class_thermo", th)

        # --- splines --- #

        rho = th["rho"].value
        g = th["g"].to_value(1 / u.Mpc).copy()
        g.flags.writeable = True

        # ext 1 makes stuff 0, which is correct, but introduces a small discontinuity.
        self._spl_gbarCL: IUSpline
        object.__setattr__(self, "_spl_gbarCL", IUSpline(rho, g, ext=1))

        # For the log we need to avoid log(0), and also avoid a discontinuity,
        # and ensure that the extrapolation is reasonable. Ext 3 returns the bou
        # value, which is what we want.
        g = g.copy()
        g[g <= 0] = 1e-300  # should be small enough that discontinuity is negligible
        self._spl_ln_gbarCL: IUSpline
        object.__setattr__(self, "_spl_ln_gbarCL", IUSpline(rho, np.log(g), ext=3))

        # Instead, to get the right normalization we will define PbarCL from an integral
        # over gbarCL.
        integral = [self._spl_gbarCL.integral(a, b) for a, b in itertools.pairwise(rho)]
        PbarCL = self.lambda0.value * np.concatenate(([0], np.cumsum(integral)))
        _spl_PbarCL = IUSpline(rho, PbarCL, ext=2)

        # normalize the spline
        a, b = self.rho_domain
        norm = self.lambda0.value * self._spl_gbarCL.integral(a, b) / _spl_PbarCL(b)
        self._spl_integral_norm: float
        object.__setattr__(self, "_spl_integral_norm", 1 / norm)

        norm_PbarCL = PbarCL / norm
        self._spl_PbarCL: IUSpline
        object.__setattr__(self, "_spl_PbarCL", IUSpline(rho, norm_PbarCL, ext=2))

        # For the log we need to avoid log(0) and also avoid a discontinuity
        norm_PbarCL[0] = norm_PbarCL[1]  # known to be 0, avoid log(0)
        lnPbarCL = np.log(norm_PbarCL)
        lnPbarCL[0] = lnPbarCL[1] - (lnPbarCL[2] - lnPbarCL[1])  # assign by extrapolate
        self._spl_ln_PbarCL: IUSpline
        object.__setattr__(self, "_spl_ln_PbarCL", IUSpline(rho, lnPbarCL, ext=2))

        # --- assistive --- #

        self.P: ComputePspllSprp
        object.__setattr__(
            self,
            "P",
            ComputePspllSprp(
                lambda0=float(self.lambda0.to_value(u.Mpc)),
                rho_domain=(
                    float(self.rho_domain[0].to_value(u.one)),
                    float(self.rho_domain[1].to_value(u.one)),
                ),  # TODO! confirm (min, max)
                spl_ln_gbarCL=self._spl_ln_gbarCL,
                spl_ln_PbarCL=self._spl_ln_PbarCL,
            ),
        )

    @property
    def class_thermo(self) -> MappingProxyType[str, Any]:
        """CLASS thermodynamics."""
        if "class_thermo" in self.__dict__:
            ct: MappingProxyType[str, Any] = self.__dict__["class_thermo"]
        else:
            self.__dict__["class_thermo"] = ct = MappingProxyType(self._class_thermo)
        return ct

    @property
    def lambda0(self) -> Quantity:
        """Distance scale factor in Mpc."""
        return self.distance_converter.lambda0.to(u.Mpc)

    # ===============================================================
    # Properties

    @cached_property
    def z_domain(self) -> tuple[Quantity, Quantity]:
        """Z domain of validity."""
        return (self._class_thermo["z"][0], self._class_thermo["z"][-1])

    @cached_property
    def rho_domain(self) -> tuple[Quantity, Quantity]:
        """Rho domain of validity."""
        return (self._class_thermo["rho"][0], self._class_thermo["rho"][-1])
        # return (

    @property
    def maxrho_domain(self) -> Quantity:
        """Maximum rho domain of validity."""
        return self.rho_domain[1]

    # --------

    @property
    def z_recombination(self) -> Quantity:
        """Z of peak of visibility function."""
        return self._class_thermo["z"][self._class_thermo["g"].argmax()] << cu.redshift

    @property
    def rho_recombination(self) -> Quantity:
        """Rho of peak of visibility function."""
        return self._class_thermo["rho"][self._class_thermo["g"].argmax()] << u.one

    # ===============================================================

    def plot_CLASS_points_distribution(
        self, *, density: bool = False, figure_kw: Mapping[str, Any] | None = None
    ) -> Figure:
        """Plot the distribution of evaluation points."""
        fig = plt.figure(**(figure_kw or {}))
        ax = fig.add_subplot()

        z = cast("Quantity", self._class_thermo["z"])

        ax.hist(np.log10(z.value), density=density, bins=30)
        ax.axvline(
            np.log10(self.z_recombination.to_value(cu.redshift)),
            c="k",
            ls="--",
            label="Recombination",
        )

        ax.set_xlabel(r"$log_{10}(z)$")
        ax.set_ylabel("frequency" + ("density" if density else ""))
        ax.set_title("CLASS points distribution")
        ax.legend()

        return fig

    def plot_zv_choice(self) -> Figure:
        """Plot ``z_v`` choice."""
        fig = plt.figure(figsize=(10, 4))

        zs = self._class_thermo["z"]
        gs = self._class_thermo["g"]
        rhos = self._class_thermo["rho"]

        # --------------
        # g vs z

        ax1 = fig.add_subplot(
            121,
            title=r"Choosing $z_V$",
            xlabel="z",
            ylabel=r"$g_\gamma^{CL}$",
        )

        # z-valid
        _zVi, zVf = self.z_domain
        zVi = None if _zVi == zs[0] else _zVi
        gbarCL_O = gs[np.argmin(np.abs(zs - zVf))]

        ind = gs > 1e-18 / u.Mpc  # cutoff for plot clarity
        ax1.loglog(zs[ind], gs[ind])
        if zVi is not None:
            ax1.axvline(zVi, c="k", ls=":", label=rf"$z$={zVi}")
        ax1.axvline(zVf, c="k", ls=":", label=rf"$z_V$={zVf}")

        ax1.axhline(
            gbarCL_O.value,
            c="k",
            ls=":",
            label=r"$g_\gamma^{CL}(z_V)$=" + f"{gbarCL_O:.2e}",  # noqa: ISC003
        )
        ax1.invert_xaxis()
        ax1.legend()

        # --------------
        # g vs rho

        ax2 = fig.add_subplot(
            122,
            xlabel=r"$\rho$",
            ylabel=r"$g_\gamma^{CL}(\rho)$",
        )

        rho_o0, rho_o1 = self.rho_domain

        ax2.plot(rhos[ind], gs[ind])
        if zVi is not None:
            rho_o0 = self.distance_converter.rho_of_z(zVi)
            ax2.axvline(rho_o0.value, c="k", ls=":", label=rf"$\rho_i$={rho_o0:.2e}")
        ax2.axvline(rho_o1.value, c="k", ls=":", label=rf"$\rho_V$={rho_o1:.2e}")
        ax2.axhline(
            gbarCL_O.value,
            c="k",
            ls=":",
            label=r"$g_\gamma^{CL}(\rho_V)$=" + f"{gbarCL_O:.2e}",  # noqa: ISC003
        )
        ax2.set_yscale("log")
        ax2.legend()

        fig.tight_layout()
        return fig
