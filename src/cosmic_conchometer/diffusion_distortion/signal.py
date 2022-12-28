"""Diffusion Damping Distortion."""

from __future__ import annotations

# STDLIB
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

# THIRD-PARTY
import numpy as np

# LOCAL
from cosmic_conchometer.ps import PowerSpectrumCallable
from cosmic_conchometer.tf import TransferFunctionCallable

if TYPE_CHECKING:
    # LOCAL
    from cosmic_conchometer._typing import NDAf, scalarT
    from cosmic_conchometer.params import LCDMParameters


__all__: list[str] = []


class FFTPCallable(Protocol):
    """Protocol for FFT of P."""

    def __call__(
        self, x: scalarT | NDAf, y: scalarT | NDAf, *, grid: bool
    ) -> scalarT | NDAf:
        """FFT of P."""
        ...


@dataclass(frozen=True)
class SpectralDistortionSignalIntegrand:
    """Thing."""

    fftP: FFTPCallable
    ps: PowerSpectrumCallable
    tf: TransferFunctionCallable

    def power_transfer2(
        self,
        cosmo_params: LCDMParameters,
        kmag: scalarT | NDAf,
        *,
        pivot_scale: scalarT,
        z_last_scatter: scalarT,
    ) -> scalarT | NDAf:
        """Power transfer function."""
        return (
            self.ps(cosmo_params, kmag, pivot_scale=pivot_scale)
            * self.tf(cosmo_params, kmag, z_last_scatter=z_last_scatter) ** 2
        )

    def __call__(
        self,
        cosmo_params: LCDMParameters,
        kpll: scalarT | NDAf,
        kprp: scalarT | NDAf,
        *,
        pivot_scale: float = 5e-2,
        z_last_scatter: float = 1_100,
    ) -> scalarT | NDAf:
        """Spectral distortion signal integrand.

        Parameters
        ----------
        cosmo_params : LCDMParameters
            Cosmological parameters.
        kpll, kprp : ndarray
            Units of inverse Mpc.

        pivot_scale : float, optional
            Units of Mpc.
        z_last_scatter : float, optional
            Units of Mpc.

        Returns
        -------
        ndarray
            Units of Mpc^3.

        Notes
        -----
        For `scipy.integrate.dblquad` ordered y, x.
        Need to multiply integral by pi * As
        """
        kmag = np.sqrt(kpll**2 + kprp**2)
        term1 = kprp * self.power_transfer2(
            cosmo_params, kmag, pivot_scale=pivot_scale, z_last_scatter=z_last_scatter
        )
        return term1 * (1 - self.fftP(kpll, kprp, grid=False) ** 2)
