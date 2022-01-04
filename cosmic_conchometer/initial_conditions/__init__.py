# -*- coding: utf-8 -*-

"""Initial Conditions."""


##############################################################################
# IMPORTS

# STDLIB
import typing as T

# THIRD PARTY
import astropy.units as u
import numpy as np
from scipy.stats import norm as Gaussian, uniform

__all__ = ["TemperatureFluctuationsFourierSpace"]

##############################################################################


class TemperatureFluctuationsFourierSpace:
    """Draw a realization of the temperature field in k-space.

    Parameters
    ----------
    nx, ny, nz : int, keyword-only
        Maximum value of the Cartesian components :math:`n = \frac{L}{2\pi} k`.
    step : int, optional keyword-only
        Step-size in making the n grid [-n, n], default 10.
    L : |Quantity|, optional keyword-only
        Size of the periodic box. Default is 30 Gpc.
    As : float, optional keyword-only
        Scalar amplitude. See Planck paper [1]_.
    ns : float, optional keyword-only
        Scalar spectral index. See Planck paper [1]_.
    pivot_scale : |Quantity|, optional keyword-only
        The scale at which ``ns`` is normalized. See Planck paper [1]_.

    References
    ----------
    .. [1] Planck Collaboration, et al (2020). Planck 2018 results. VI.
        Cosmological parameters. \aap, 641, A6.
    """

    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        nz: int,
        step: int = 10,
        L: u.Quantity = 30 * u.Gpc,
        As: float = 1e-10 * np.exp(3.04),
        ns: float = 0.96,
        pivot_scale: u.Quantity = 0.05 / u.Mpc
    ) -> None:
        L = L.to_value(u.Mpc)

        # make meshgrid of n
        nxs = np.arange(-nx, nx + 1, step=step, dtype=int)
        nys = np.arange(-ny, ny + 1, step=step, dtype=int)
        nzs = np.arange(-nz, nz + 1, step=step, dtype=int)

        nX, nY, nZ = np.meshgrid(nxs, nys, nzs, indexing="ij")

        self._nzs = nzs
        self._sel_upper = nzs >= 0
        self._shape = nZ.shape
        self._shape_upper = (*nZ.shape[:-1], len(self._sel_upper[self._sel_upper]))

        # kvec (inv Mpc) over the mesh
        kX = 2 * np.pi / L * nX
        kY = 2 * np.pi / L * nY
        kZ = 2 * np.pi / L * nZ
        kmag = np.sqrt(kX ** 2 + kY ** 2 + kZ ** 2)

        # don't care about k=0
        self._mask = np.where(kmag == 0)
        kmag[self._mask] = np.nan

        # Power spectrum (mesh)
        Pk = As * np.power(kmag / pivot_scale.to_value(1 / u.Mpc), ns - 1)
        variance = 2 * np.pi ** 2 / (kmag * L) ** 3 * Pk
        variance[self._mask] = 0
        self._std = np.sqrt(variance)

    def __call__(
        self, *, random_state: T.Union[None, int, np.random.Generator, np.random.RandomState] = None
    ) -> np.ndarray:
        """Draw a realization of the temperature fluctuation field in k-space.

        Parameters
        ----------
        random_state : {None, int, `numpy.random.Generator`,
                `numpy.random.RandomState`}, optional keyword-only

            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.

        Returns
        -------
        realization : ndarray
            The complex-valued temperature field in k-space.
            A mesh of shape
        """
        # Random amplitude. A Gaussian of mean=0, sigma=1. It has point parity around the kz origin.
        amp = np.empty(self._shape)
        amp[:, :, self._sel_upper] = Gaussian.rvs(
            loc=0, scale=1, size=self._shape_upper, random_state=random_state
        )
        amp[:, :, ~self._sel_upper] = amp[:, :, self._sel_upper][:, :, 1:][
            :, :, ::-1
        ]  # enforce point parity
        # ↑ fills the kz<0 by taking  (               kz > 0             ) and flipping.
        amp[self._mask] = 0  # enforce kx=ky=kz=0 is "ignored"

        # Generate the phase. It has point anti-symmetry around the kz origin.
        phase = np.empty(self._shape)
        phase[:, :, self._sel_upper] = uniform.rvs(
            loc=0, scale=2 * np.pi, size=self._shape_upper, random_state=random_state
        )
        phase[:, :, ~self._sel_upper] = -phase[:, :, self._sel_upper][:, :, 1:][
            :, :, ::-1
        ]  # enforce point anti-symmetry

        realization = (np.cos(phase) + 1j * np.sin(phase)) * amp * self._std
        return realization
