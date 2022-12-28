"""Core functions safe to be used in lower modules."""

from __future__ import annotations

# STDLIB
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, TypeVar

# THIRD-PARTY
import astropy.cosmology.units as cu
import astropy.units as u
from astropy.cosmology import LambdaCDM, Planck18

# LOCAL
from cosmic_conchometer.utils.distances import z_matter_radiation_equality as _calc_zeq

if TYPE_CHECKING:
    # THIRD-PARTY
    from astropy.units import Quantity

__all__ = ["CosmologyParameters", "LCDMParameters", "planck18"]

Self = TypeVar("Self", bound="LCDMParameters")


##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class CosmologyParameters:
    """Cosmology Parameters."""

    def __post_init__(self) -> None:
        """Post-initialization."""


@dataclass(frozen=True)
class LCDMParameters(CosmologyParameters):
    """FLRW Parameters."""

    h: float
    Om0: float
    Ode0: float
    Tcmb0: float  # units of K
    Neff: float
    Ob0: float
    m_nu: tuple[float, ...]  # units of eV
    As: float
    ns: float

    def __post_init__(self) -> None:
        self.cosmo: LambdaCDM

        # TODO: use FlatLambdaCDM if appropriate
        object.__setattr__(
            self,
            "cosmo",
            LambdaCDM(
                H0=100 * self.h * (u.km / u.s / u.Mpc),
                Om0=self.Om0,
                Ode0=self.Ode0,
                Tcmb0=self.Tcmb0 * u.K,
                Neff=self.Neff,
                Ob0=self.Ob0,
                m_nu=self.m_nu * u.eV,
            ),
        )

    @classmethod
    def from_astropy(  # noqa: D417
        cls: type[Self], cosmo: LambdaCDM, /, As: float, ns: float
    ) -> Self:
        """From an Astropy cosmology.

        Parameters
        ----------
        cosmo : `~astropy.cosmology.LambdaCDM`
            The astropy cosmology.
        As : float
            The amplitude of the primordial power spectrum.
        ns : float
            The spectral index of the primordial power spectrum.

        Returns
        -------
        `~cosmic_conchometer.params.LCDMParameters`
        """
        return cls(
            h=float(cosmo.h),
            Om0=float(cosmo.Om0),
            Ode0=float(cosmo.Ode0),
            Tcmb0=float(cosmo.Tcmb0.to_value("K")),
            Neff=float(cosmo.Neff),
            Ob0=float(cosmo.Ob0),
            m_nu=tuple(cosmo.m_nu.to_value("eV")),
            As=As,
            ns=ns,
        )

    @cached_property
    def z_matter_radiation_equality(self) -> Quantity:
        """matter-radiation equality redshift."""
        return _calc_zeq(self.cosmo, full_output=False) << cu.redshift


# TODO: As
planck18 = LCDMParameters.from_astropy(Planck18, As=1, ns=0.972)
