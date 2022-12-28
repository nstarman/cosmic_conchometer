"""Intrinsic Distortion Core Functions."""

from __future__ import annotations

# STDLIB
import itertools
from collections.abc import Mapping
from dataclasses import InitVar, dataclass
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, cast

# THIRD-PARTY
import astropy.cosmology.units as cu
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from classy import Class as Classy
from scipy.interpolate import InterpolatedUnivariateSpline as IUSpline

# LOCAL
from cosmic_conchometer.common import CosmologyDependent
from cosmic_conchometer.diffusion_distortion.prob_2ls import ComputePspllSprp

if TYPE_CHECKING:
    # THIRD-PARTY
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
class SpectralDistortion(CosmologyDependent):
    r"""Spectral Distortion.

    Parameters
    ----------
    cosmo : `~astropy.cosmology.core.Cosmology`
    class_cosmo : :class:`classy.Class`
    AkFunc: Callable, str, or None (optional, keyword-only)
        The function to calculate :math:`A(\vec{k})`
    """

    class_cosmo: Classy
    z_bounds: InitVar[tuple[float | None, float]] = (None, 100)

    def __post_init__(self, z_bounds: tuple[float | None, float] = (None, 100)) -> None:
        super().__post_init__()

        # --- thermodynamic quantities --- #

        th = self.class_cosmo.get_thermodynamics()

        # time-order everything (inplace)
        for k, v in th.items():
            th[k] = v[::-1]
            th[k].flags.writeable = False

        # now that have z array, can fix unbounded min(z_domain)
        if z_bounds[0] is None:
            z_domain = (th["z"][0], z_bounds[1])
        else:
            z_domain = z_bounds
        self.z_domain: tuple[float, float]
        object.__setattr__(self, "z_domain", z_domain)

        # rename and assign units
        it: tuple[tuple[str, str, u.UnitBase], ...] = (
            ("z", "z", cu.redshift),
            ("g [Mpc^-1]", "g", u.Unit("1 / Mpc")),
        )
        for name, newname, unit in it:
            th[newname] = th.pop(name) * unit

        # derived values
        th["rho"] = self.distance_converter.rho_of_z(th["z"])

        # make everything immutable again, b/c added stuff
        for k in th.keys():
            th[k].flags.writeable = False

        self._class_thermo: Mapping[str, NDArray[np.floating[Any]]]
        object.__setattr__(self, "_class_thermo", th)

        # --- splines --- #

        rho = th["rho"]

        self._spl_gbarCL: IUSpline
        object.__setattr__(self, "_spl_gbarCL", IUSpline(rho, th["g"], ext=1))
        # ext 1 makes stuff 0  # TODO! does this introduce a discontinuity?

        # Instead, to get the right normalization we will define PbarCL from an integral
        # over gbarCL.
        integral = [self._spl_gbarCL.integral(a, b) for a, b in itertools.pairwise(rho)]
        PbarCL = self.lambda0.value * np.concatenate(([0], np.cumsum(integral)))
        _spl_PbarCL = IUSpline(rho, PbarCL, ext=2)

        # normalize the spline
        a, b = self.rho_domain
        prob = self.lambda0.value * self._spl_gbarCL.integral(a, b) / _spl_PbarCL(b)
        self._spl_integral_norm: float
        object.__setattr__(self, "_spl_integral_norm", 1 / prob)

        self._spl_PbarCL: IUSpline
        object.__setattr__(self, "_spl_PbarCL", IUSpline(rho, PbarCL / prob, ext=2))

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
                spl_gbarCL=self._spl_gbarCL,
                spl_PbarCL=self._spl_PbarCL,
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

    # ===============================================================
    # Properties

    @cached_property
    def rho_domain(self) -> tuple[Quantity, Quantity]:
        """Rho domain of validity."""
        return (
            self.distance_converter.rho_of_z(self.z_domain[0]),
            self.distance_converter.rho_of_z(self.z_domain[1]),
        )

    @property
    def maxrho_domain(self) -> Quantity:
        """Maximum rho domain of validity."""
        return self.rho_domain[1]

    # --------

    @property
    def z_recombination(self) -> Quantity:
        """z of peak of visibility function."""
        return self._class_thermo["z"][self._class_thermo["g"].argmax()] << cu.redshift

    @property
    def rho_recombination(self) -> Quantity:
        """rho of peak of visibility function."""
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
            label=r"$g_\gamma^{CL}(z_V)$=" f"{gbarCL_O:.2e}",
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
            label=r"$g_\gamma^{CL}(\rho_V)$=" f"{gbarCL_O:.2e}",
        )
        ax2.set_yscale("log")
        ax2.legend()

        fig.tight_layout()
        return fig
