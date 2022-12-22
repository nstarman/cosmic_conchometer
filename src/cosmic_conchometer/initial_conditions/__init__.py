
"""Initial Conditions."""


##############################################################################
# IMPORTS

# STDLIB
import typing as T


import astropy.units as u
import numpy as np
from scipy.stats import norm as Gaussian
from scipy.stats import uniform

__all__ = ["TemperatureFluctuationsFourierSpace"]

##############################################################################


def kmag_from_ns(
    nx: int,
    ny: int,
    nz: int,
    step: int = 10,
    Lbox: u.Quantity = 30 * u.Gpc,
) -> u.Quantity:
    """Make (N, M, L) ndarray of :math:`|k|`.

    Parameters
    ----------
    nx, ny, nz : int, keyword-only
        Maximum value of the Cartesian components :math:`n = \frac{L}{2\pi} k`.
    step : int, optional keyword-only
        Step-size in making the n grid [-n, n], default 10.
    Lbox : |Quantity|, optional keyword-only
        Size of the periodic box. Default is 30 Gpc.

    Returns
    -------
    (N, M, L) ndarray
    """
    Lbox = Lbox.to(u.Mpc)

    # make meshgrid of n
    nxs = np.arange(-nx, nx + 1, step=step, dtype=int)
    nys = np.arange(-ny, ny + 1, step=step, dtype=int)
    nzs = np.arange(-nz, nz + 1, step=step, dtype=int)

    nX, nY, nZ = np.meshgrid(nxs, nys, nzs, indexing="ij")

    # kvec (inv Mpc) over the mesh
    kmag = (2 * np.pi / Lbox) * np.sqrt(nX ** 2 + nY ** 2 + nZ ** 2)

    return kmag


def power_spectrum(
    kmag: u.Quantity, *, As: u.Quantity, ns: float, pivot_scale: u.Quantity
) -> np.ndarray:
    """

    Parameters
    ----------
    kmag : (N, M, L) Quantity
        Meshgrid (indexing = "ij") of :math:`|k|`.
        Units of inverse distance.
    As : Quantity, optional keyword-only
        Scalar amplitude. See Planck paper [1]_. Units of temperature squared.
    ns : float, optional keyword-only
        Scalar spectral index. See Planck paper [1]_.
    pivot_scale : |Quantity|, optional keyword-only
        The scale at which ``ns`` is normalized. See Planck paper [1]_.

    References
    ----------
    .. [1] Planck Collaboration, et al (2020). Planck 2018 results. VI.
        Cosmological parameters. \aap, 641, A6.
    """
    Pk = As * np.power((kmag / pivot_scale).to_value(u.one), ns - 1)
    return Pk


class TemperatureFluctuationsFourierSpace:
    """Draw a realization of the temperature field in k-space.

    Parameters
    ----------
    kmag : (N, M, L) Quantity
        Meshgrid (indexing = "ij") of :math:`|k|`.
        Units of inverse distance.
    Lbox : |Quantity|, optional keyword-only
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

    @classmethod
    def make_from_ns(
        cls,
        *,
        nx: int,
        ny: int,
        nz: int,
        step: int = 10,
        Lbox: u.Quantity = 30 * u.Gpc,
        As: u.Quantity = 1e-10 * np.exp(3.04) * u.K ** 2,
        ns: float = 0.96,
        pivot_scale: u.Quantity = 0.05 / u.Mpc
    ) -> "TemperatureFluctuationsFourierSpace":
        """Draw a realization of the temperature field in k-space.

        Parameters
        ----------
        nx, ny, nz : int, keyword-only
            Maximum value of the Cartesian components :math:`n = \frac{L}{2\pi} k`.
        step : int, optional keyword-only
            Step-size in making the n grid [-n, n], default 10.
        Lbox : |Quantity|, optional keyword-only
            Size of the periodic box. Default is 30 Gpc.
        As : float, optional keyword-only
            Scalar amplitude. See Planck paper [1]_.
        ns : float, optional keyword-only
            Scalar spectral index. See Planck paper [1]_.
        pivot_scale : |Quantity|, optional keyword-only
            The scale at which ``ns`` is normalized. See Planck paper [1]_.

        Returns
        -------
        TemperatureFluctuationsFourierSpace instance

        References
        ----------
        .. [1] Planck Collaboration, et al (2020). Planck 2018 results. VI.
            Cosmological parameters. \aap, 641, A6.
        """
        kmag = kmag_from_ns(nx=nx, ny=ny, nz=nz, step=step, Lbox=Lbox)

        inst = super().__new__(cls)
        inst.__init__(
            kmag,
            Lbox=Lbox << u.Mpc,
            As=As,
            ns=ns,
            pivot_scale=pivot_scale,
        )
        return inst

    def __init__(
        self,
        kmag,
        *,
        Lbox: u.Quantity = 30 * u.Gpc,
        As: float = 1e-10 * np.exp(3.04),
        ns: float = 0.96,
        pivot_scale: u.Quantity = 0.05 / u.Mpc
    ) -> None:
        self._shape = kmag.shape

        #  nzs >= 0
        center = np.where(kmag == 0)
        if not center:
            raise ValueError("kmag must have center = 0, indexed 'ij'.")
        sel_upper = np.zeros(kmag.shape[-1], dtype=bool)
        sel_upper[slice(int(center[-1]), None)] = True
        self._sel_upper = sel_upper
        self._shape_upper = (
            *kmag.shape[:-1],
            len(self._sel_upper[self._sel_upper]),
        )

        # don't care about k=0
        self._mask = np.where(kmag == 0)
        kmag[self._mask] = np.nan

        # Power spectrum (mesh)
        Pk = power_spectrum(kmag, As=As, ns=ns, pivot_scale=pivot_scale)

        variance = (
            2 * np.pi ** 2 / (kmag * Lbox).to_value(u.one) ** 3 * Pk.value
        )
        variance[self._mask] = 0
        self._std = np.sqrt(variance) << np.sqrt(Pk.unit)

    def __call__(
        self,
        *,
        random_state: T.Union[
            None,
            int,
            np.random.Generator,
            np.random.RandomState,
        ] = None
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
            loc=0,
            scale=1,
            size=self._shape_upper,
            random_state=random_state,
        )
        amp[:, :, ~self._sel_upper] = amp[:, :, self._sel_upper][:, :, 1:][
            ::-1,
            ::-1,
            ::-1,
        ]
        # ↑ fills the kz<0 by taking  (               kz > 0             ) and flipping.
        amp[self._mask] = 0  # enforce kx=ky=kz=0 is "ignored"

        # Generate the phase. It has point anti-symmetry around the kz origin.
        phase = np.empty(self._shape)
        phase[:, :, self._sel_upper] = uniform.rvs(
            loc=0,
            scale=2 * np.pi,
            size=self._shape_upper,
            random_state=random_state,
        )
        phase[:, :, ~self._sel_upper] = -phase[:, :, self._sel_upper][
            :,
            :,
            1:,
        ][::-1, ::-1, ::-1]

        realization = (np.cos(phase) + 1j * np.sin(phase)) * amp * self._std
        return realization
