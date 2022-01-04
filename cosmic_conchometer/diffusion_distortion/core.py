# -*- coding: utf-8 -*-

"""Intrinsic Distortion Core Functions."""

__all__ = [
    "DiffusionDistortionBase",
]


##############################################################################
# IMPORTS

# STDLIB
import typing as T
from abc import abstractmethod
from types import MappingProxyType

# THIRD PARTY
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology.core import Cosmology
from astropy.utils.decorators import lazyproperty
from classy import Class as CLASS

# PROJECT-SPECIFIC
from cosmic_conchometer.common import CosmologyDependent, default_Ak
from cosmic_conchometer.typing import ArrayLike, ArrayLikeCallable
from cosmic_conchometer.utils import distances

##############################################################################
# PARAMETERS

IUSType = T.Callable[[ArrayLike], np.ndarray]


##############################################################################
# CODE
##############################################################################


class DiffusionDistortionBase(CosmologyDependent):
    r"""Spectral Distortion.

    Parameters
    ----------
    cosmo : `~astropy.cosmology.core.Cosmology`
    class_cosmo : :class:`classy.Class`
    AkFunc: Callable, str, or None (optional, keyword-only)
        The function to calculate :math:`A(\vec{k})`

    """

    _zvalid: T.Tuple[float, float]

    def __init__(
        self,
        cosmo: Cosmology,
        class_cosmo: CLASS,
        zvalid: T.Tuple[T.Optional[float], float] = (None, 100),
        *,
        AkFunc: T.Union[str, ArrayLikeCallable, None] = None,
        **kwargs: T.Any,
    ) -> None:
        super().__init__(cosmo)
        self.class_cosmo = class_cosmo  # TODO? move to superclass

        # --- Ak --- #
        self.AkFunc: ArrayLikeCallable
        if AkFunc is None:
            self.AkFunc = default_Ak.get()
        elif isinstance(AkFunc, str) or callable(AkFunc):
            with default_Ak.set(AkFunc):
                self.AkFunc = default_Ak.get()
        else:
            raise TypeError("AkFunc must be <None, str, callable>.")

        # --- thermodynamic quantities --- #

        th = class_cosmo.get_thermodynamics()
        self._class_thermo_ = th  # editible version
        self._class_thermo = MappingProxyType(self._class_thermo_)

        # time-order everything (inplace)
        for k, v in th.items():
            th[k] = v[::-1]
            th[k].flags.writeable = False

        # now that have z array, can fix unbounded min(zvalid)
        if zvalid[0] is None:
            _zvalid = (th["z"][0], zvalid[1])
        self._zvalid = _zvalid

        # rename and assign units
        it: T.Tuple[T.Tuple[str, str, u.Unit], ...] = (
            ("conf. time [Mpc]", "conformal time", u.Mpc),
            ("kappa' [Mpc^-1]", "kappa'", 1 / u.Mpc),
            ("exp(-kappa)", "exp(-kappa)", 1),
            ("g [Mpc^-1]", "g", 1 / u.Mpc),
            ("Tb [K]", "Tb", u.K),
            ("dTb [K]", "dTb", u.K),
        )
        for name, newname, unit in it:
            th[newname] = th.pop(name) * unit

        # derived values
        zeq = self._zeq = distances.z_of.matter_radiation_equality(cosmo=cosmo)
        rho = th["rho"] = distances.rho_of.z(th["z"], zeq=zeq)

        # make everything immutable again, b/c added stuff
        for k in th.keys():
            th[k].flags.writeable = False

    # ===============================================================

    @property
    def zvalid(self) -> T.Tuple[float, float]:
        return self._zvalid

    @lazyproperty
    def rhovalid(self) -> T.Tuple[float, float]:
        return (
            distances.rho_of.z(self.zvalid[0], zeq=self._zeq),
            distances.rho_of.z(self.zvalid[1], zeq=self._zeq),
        )

    @lazyproperty
    def maxrhovalid(self) -> float:
        rho: float = self.rhovalid[1]
        return rho

    # --------

    @lazyproperty
    def rho_eq(self) -> float:
        rho: float = distances.rho_of.matter_radiation_equality  # := 1
        return rho

    # --------

    @lazyproperty
    def z_recombination(self) -> float:
        """z of peak of visibility function."""
        z: float = self.class_z[self.class_g.argmax()]
        return z

    @lazyproperty
    def rho_recombination(self) -> float:
        """rho of peak of visibility function."""
        rho: float = self.class_rho[self.class_g.argmax()]
        return rho

    # --------------------------------------------
    # CLASS properties

    @property
    def class_thermo(self) -> MappingProxyType:
        return self._class_thermo

    @property
    def class_z(self) -> np.ndarray:
        z: np.ndarray = self._class_thermo["z"]
        return z

    @property
    def class_eta(self) -> np.ndarray:
        eta: np.ndarray = self._class_thermo["conformal time"]
        return eta

    @property
    def class_dk_dt(self) -> np.ndarray:
        dkdt: np.ndarray = self._class_thermo["kappa'"]
        return dkdt

    @property
    def class_P(self) -> np.ndarray:
        """:math:`\bar{P_\gamma}^{CL}`"""
        P: np.ndarray = self._class_thermo["exp(-kappa)"]
        return P

    @property
    def class_g(self) -> np.ndarray:
        g: np.ndarray = self._class_thermo["g"]
        return g

    @property
    def class_rho(self) -> np.ndarray:
        rho: np.ndarray = self._class_thermo["rho"]
        return rho

    @lazyproperty
    def class_rhomin(self) -> float:
        rho: float = min(self.class_rho)
        return rho

    @lazyproperty
    def class_rhomax(self) -> float:
        rho: float = max(self.class_rho)
        return rho

    # ===============================================================

    @abstractmethod
    def __call__(self) -> u.Quantity:
        """Perform computation."""
        pass

    # ===============================================================
    # Convenience methods

    def set_AkFunc(self, value: T.Union[None, str, T.Callable]) -> T.Callable:
        """Set the default function used in A(k).

        Can be used as a contextmanager, same as `~.default_Ak`.

        Parameters
        ----------
        value

        Returns
        -------
        `~astropy.utils.state.ScienceStateContext`
            Output of :meth:`~.default_Ak.set`

        """
        self.Akfunc: T.Callable = default_Ak.set(value)
        return self.Akfunc

    # ===============================================================

    def plot_CLASS_points_distribution(self, density: bool = False) -> plt.Figure:
        """Plot the distribution of evaluation points."""
        fig = plt.figure()
        ax = fig.add_subplot()

        ax.hist(np.log10(self.class_z[:-1]), density=density, bins=30)
        ax.axvline(
            np.log10(self.z_recombination),
            c="k",
            ls="--",
            label="Recombination",
        )

        ax.set_xlabel(r"$log_{10}(z)$")
        ax.set_ylabel("frequency" + ("density" if density else ""))
        ax.set_title("CLASS points distribution")
        ax.legend()

        return fig

    def plot_zv_choice(self) -> plt.Figure:
        """Plot ``z_v`` choice."""
        fig = plt.figure(figsize=(10, 4))

        # --------------
        ax1 = fig.add_subplot(121, title=r"Choosing $z_V$", xlabel="z", ylabel=r"$g_\gamma^{CL}$")

        # z-valid
        _zVi, zVf = self.zvalid
        zVi = None if _zVi == self.class_z[0] else _zVi
        gbarCL_V = self.class_g[np.argmin(np.abs(self.class_z - zVf))]

        ind = self.class_g > 1e-18 / u.Mpc  # cutoff for plot clarity
        ax1.loglog(self.class_z[ind], self.class_g[ind])
        if zVi is not None:
            ax1.axvline(zVi, c="k", ls=":", label=fr"$z$={zVi}")
        ax1.axvline(zVf, c="k", ls=":", label=fr"$z_V$={zVf}")

        ax1.axhline(gbarCL_V.value, c="k", ls=":", label=r"$g_\gamma^{CL}(z_V)$=" f"{gbarCL_V:.2e}")
        ax1.invert_xaxis()
        ax1.legend()

        # --------------
        ax2 = fig.add_subplot(122, xlabel=r"\rho", ylabel=r"$g_\gamma^{CL}(\rho)$")

        rhoVi, rhoVf = self.rhovalid

        ax2.plot(self.class_rho[ind], self.class_g[ind])
        if zVi is not None:
            rhoVi = distances.rho_of.z(zVi, zeq=self.z_eq)
            ax2.axvline(rhoVi, c="k", ls=":", label=fr"$\rho_i$={rhoVi:.2e}")
        ax2.axvline(rhoVf, c="k", ls=":", label=fr"$\rho_V$={rhoVf:.2e}")
        ax2.axhline(
            gbarCL_V.value, c="k", ls=":", label=r"$g_\gamma^{CL}(\rho_V)$=" f"{gbarCL_V:.2e}"
        )
        ax2.set_yscale("log")
        ax2.legend()

        return fig


##############################################################################
# END
