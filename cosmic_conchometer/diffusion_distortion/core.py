# -*- coding: utf-8 -*-

"""Intrinsic Distortion Core Functions."""

__all__ = [
    "DiffusionDistortionBase",
]


##############################################################################
# IMPORTS

# BUILT-IN
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

    def __init__(
        self,
        cosmo: Cosmology,
        class_cosmo: CLASS,
        zvalid=(None, 100),  # TODO! auto-determine if None
        *,
        AkFunc: T.Union[str, ArrayLikeCallable, None] = None,
        **kwargs
    ) -> None:
        super().__init__(cosmo)
        self.class_cosmo = class_cosmo  # TODO? move to superclass
        self._zvalid = zvalid

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
        if self._zvalid[0] is None:
            self._zvalid = (th["z"][0], self._zvalid[1])

        # rename and assign units
        it = (
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

    # /def

    # ===============================================================

    @property
    def zvalid(self):
        return self._zvalid

    @lazyproperty
    def rhovalid(self):
        return (
            distances.rho_of.z(self.zvalid[0], zeq=self._zeq),
            distances.rho_of.z(self.zvalid[1], zeq=self._zeq),
        )

    @lazyproperty
    def maxrhovalid(self):
        return self.rhovalid[1]

    # --------

    @property
    def z_eq(self):
        return self._zeq

    @lazyproperty
    def rho_eq(self):
        return distances.rho_of.matter_radiation_equality  # := 1

    # --------

    @lazyproperty
    def z_recombination(self):
        """z of peak of visibility function."""
        return self.class_z[self.class_g.argmax()]

    @lazyproperty
    def rho_recombination(self):
        """rho of peak of visibility function."""
        return self.class_rho[self.class_g.argmax()]

    # --------------------------------------------
    # CLASS properties

    @property
    def class_thermo(self):
        return self._class_thermo

    @property
    def class_z(self):
        return self._class_thermo["z"]

    @property
    def class_eta(self):
        return self._class_thermo["conformal time"]

    @property
    def class_dk_dt(self):
        return self._class_thermo["kappa'"]

    @property
    def class_P(self):
        """:math:`\bar{P_\gamma}^{CL}`"""
        return self._class_thermo["exp(-kappa)"]

    @property
    def class_g(self):
        return self._class_thermo["g"]

    @property
    def class_rho(self):
        return self._class_thermo["rho"]

    @lazyproperty
    def class_rhomin(self):
        return min(self.class_rho)

    @lazyproperty
    def class_rhomax(self):
        return max(self.class_rho)

    # ===============================================================

    @abstractmethod
    def __call__(self) -> u.Quantity:
        """Perform computation."""
        pass

    # /def

    # ===============================================================
    # Convenience methods

    @staticmethod
    def set_AkFunc(value: T.Union[None, str, T.Callable]) -> T.Callable:
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

    # /def

    # -----------------------------------------------------

    def plot_class_points_distribution(self, density=False):
        fig = plt.figure()
        ax = fig.add_subplot()

        ax.hist(np.log10(self.class_z[:-1]), density=density, bins=30)
        ax.set_xlabel(r"$log_{10}(z)$")
        ax.set_ylabel("frequency" + ("density" if density else ""))
        ax.set_title("CLASS points distribution")

        return fig

    # /def


# /class


##############################################################################
# END
