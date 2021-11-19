# -*- coding: utf-8 -*-

"""Transfer Function."""


##############################################################################
# IMPORTS

# STDLIB
import abc
import typing as T
from collections.abc import Mapping
from numbers import Number

# THIRD PARTY
import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy.integrate as integ
from astropy.cosmology.core import Cosmology
from numpy import cos, exp, log, power, sin, sqrt

# PROJECT-SPECIFIC
from .base import TransferFunctionBase, N
from .utils import llc_integrand
from cosmic_conchometer.common import CosmologyDependent

__all__ = ["TransferFunctionHuSugiyama1995Theta0"]

##############################################################################
# PARAMETERS

alpha1: float = 0.11
alpha2: float = 0.097
beta: float = 1.6
fnu: float = 0.45  # HS between A - 14 and A - 15 ratio of neutrino energy density to total radiation energy density

# Planck 2018 values  # TODO! this needs to be a Cosmology-Dependent
# Ob0: float = 0.04897
# h: float = 0.6766
# H0: float = 100.0 * h
# a0: float = 1 + zeq  # ince HS normalize aeq = 1
# keq: float = sqrt(2 * Ob0 * H0 ** 2 * a0)  # HS after A - 17  # FIXME! Or D-8
# Req: float = 0.75 / (1 - fnu) * (Ob0 / Om0)
# d2R: float = 0.25 * keq ** 2 * Req


##############################################################################


class TransferFunctionHuSugiyama1995Theta0(TransferFunctionBase):
    """Transfer Function from Poor Man's Boltzmann Code in Hu and Sugiyama 1995.

    Parameters
    ----------
    cosmo : `~astropy.cosmology.core.Cosmology` instance
    R : float
        The scaled scale factor :math:`R = a * R_{eq}`.
    krange : |Quantity|
        The (lower, change, upper) bound in units of inverse Megaparsecs.
        'change' is the 'k' to change between the two approximation regimes.
    knum : int
        Number of k's in each of the two approximation regimes.
    domain : (2,) Quantity or None, optional
        Domain to use for the returned series. If `None`, then a minimal domain
        that covers the points `x` is chosen.
    integ_kw : mapping or None (optional, keyword-only)
        Keyword arguments for integrations.
        See :func:`scipy.integrate.quad`.
    meta : mapping or None (optional, keyword-only)
        Metadata.

    References
    ----------
    .. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic Microwave
           Background: an Analytic Approach. \apj, 444, 489.
    """

    _R: float
    _knum: int
    _kmin: float
    _kchng: float
    _kmax: float
    _fit: np.polynomial.Polynomial
    _fit_info: T.Tuple

    def __init__(
        self,
        cosmo: Cosmology,
        R: float,
        *,
        krange: u.Quantity = (1e-4, 0.03, 10.0) / u.Mpc,
        knum: int = 100,
        domain: T.Optional[u.Quantity] = None,
        integ_kw: T.Optional[Mapping] = None,
        meta: T.Optional[Mapping] = None,
    ):
        super().__init__(cosmo=cosmo, meta=meta)

        self._integ_kw = integ_kw or {}

        # dimensionless radius  # TODO! recast as scale factor
        self._R = R
        if not np.isscalar(R):
            raise ValueError(f"`R` must be a scalar, not {type(R)}.")

        # k
        self._knum = knum
        self._kmin, self._kchng, self._kmax = krange.to_value(1 / u.Mpc)

        # evaluate lower approximation
        karr_ls = np.geomspace(self._kmin, self._kchng, knum) / u.Mpc
        ls = self.Theta0hatLS(R, karr_ls, integ_kw=integ_kw)

        # evaluate upper approximation
        karr_ss = np.geomspace(self._kchng, self._kmax, knum) / u.Mpc
        ss = self.Theta0hatSS(R, karr_ss, integ_kw=integ_kw)

        # put the results together
        x = np.log10(np.concatenate((karr_ss.value, karr_ls.value)))
        y = np.log10(np.concatenate((ss, ls)))

        # fit line
        poly1, info = np.polynomial.Polynomial.fit(
            x, y, deg=1, domain=domain.to_value(1 / u.Mpc) if domain is not None else None, full=True
        )
        self._fit = poly1
        self._fit_info = info

    @property
    def fitted(self) -> np.polynomial.Polynomial:
        """Fitted polynomial to :math:`\hat{\Theta}_0` approximations."""
        return self._fit

    @property
    def Req(self) -> float:  # TODO! units
        r""":math:`R_eq` in Hu & Sugiyama eq. D-8 [1]_.

        References
        ----------
        .. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic
               Microwave Background: an Analytic Approach. \apj, 444, 489.
        """
        Req: float = 0.75 / (1 - fnu) * (self.cosmo.Ob0 / self.cosmo.Om0)
        return Req

    @property
    def keq(self) -> u.Quantity:
        r""":math:`k_eq` [1/Mpc] in Hu & Sugiyama eq. D-8 and B-6 [1]_.

        This different than D-8 and B-6 by restoring a factor of c to get the
        correct units.

        References
        ----------
        .. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic
               Microwave Background: an Analytic Approach. \apj, 444, 489.
        """
        a0: float = (
            self.cosmo.Om0
            * self.cosmo.h ** 2
            / (4e-5 * (self.cosmo.Tcmb0.to_value(u.K) / 2.7) ** 4)
        )
        k_eq = sqrt(2 * self.cosmo.Om0 * self.cosmo.H0 ** 2 * a0) / const.c
        return k_eq.to(1 / u.Mpc)

    def Theta0hatSS(self, a: N, k: u.Quantity, *, integ_kw: T.Optional[Mapping] = None) -> N:
        r""":math:`\hat{\Theta}_0`, from Hu and Sugyiama [1]_, eqn. D-1.

        Parameters
        ----------
        a : scalar or array-like
            The scale factor.  # TODO! recast as rho
        k : |Quantity|
            In units of inverse Megaparsec.
        integ_kw : mapping or None (optional, keyword-only)
            Keyword arguments for integrations.
            See :func:`scipy.integrate.quad`.

        Returns
        -------
        scalar or array-like
            Same output type as inputs.

        References
        ----------
        .. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic
               Microwave Background: an Analytic Approach. \apj, 444, 489.
        """
        ss: N = _Theta0hatSS(
            a,
            k.to_value(1 / u.Mpc) * const.c.to_value(u.km / u.s),
            Req=self.Req,
            keq=self.keq.to_value(1 / u.Mpc) * const.c.to_value(u.km / u.s),
            h=self.cosmo.h,
            Ob0=self.cosmo.Ob0,
            Om0=self.cosmo.Om0,
            integ_kw=integ_kw or self._integ_kw,  # use default if integ_kw is None
        )
        return ss

    def Theta0hatLS(self, a: N, k: u.Quantity, *, integ_kw: T.Optional[Mapping] = None) -> N:
        """:math:`\hat{\Theta}_0`, from Hu and Sugyiama [1]_, eqn. D-6.

        Parameters
        ----------
        a : scalar or array-like
            The scale factor.  # TODO! recast as rho
        k : |Quantity|
            In units of inverse Megaparsec.
        integ_kw : mapping or None (optional, keyword-only)
            Keyword arguments for integrations.
            See :func:`scipy.integrate.quad`.

        Returns
        -------
        scalar or array-like
            Same output type as inputs.

        References
        ----------
        .. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic
               Microwave Background: an Analytic Approach. \apj, 444, 489.
        """
        ls: N = _Theta0hatLS(
            a,
            k.to_value(1 / u.Mpc) * const.c.to_value(u.km / u.s),
            Req=self.Req,
            keq=self.keq.to_value(1 / u.Mpc) * const.c.to_value(u.km / u.s),
            h=self.cosmo.h,
            Ob0=self.cosmo.Ob0,
            Om0=self.cosmo.Om0,
            integ_kw=integ_kw or self._integ_kw,  # use default if integ_kw is None
        )
        return ls

    def __call__(self, k: u.Quantity) -> N:
        """Evaluate :math:`\hat{\Theta}_0`, from Hu and Sugyiama [1]_.

        Parameters
        ----------
        k : |Quantity|
            In units of inverse Megaparsec.

        Returns
        -------
        float

        References
        ----------
        .. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic
               Microwave Background: an Analytic Approach. \apj, 444, 489.
        """
        theta0: N = 10 ** self._fit(np.log10(k.to_value(1 / u.Mpc)))
        return theta0

    # ===============================================================
    # Plot

    def plot_theta0hat(self) -> plt.Figure:
        """Plot :math:`\hat{\Theta}_0(|k|; \rho)`."""
        # evaluate lower approximation
        karr_ls = np.geomspace(self._kmin, self._kchng, self._knum) / u.Mpc
        ls = self.Theta0hatLS(self._R, karr_ls, integ_kw=self._integ_kw)

        # evaluate upper approximation
        karr_ss = np.geomspace(self._kchng, self._kmax, self._knum) / u.Mpc
        ss = self.Theta0hatSS(self._R, karr_ss, integ_kw=self._integ_kw)

        # and evaluate the spline
        x = np.concatenate((karr_ss, karr_ls)) << 1 / u.Mpc
        y = self(x)

        # plot
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.add_subplot(
            111,
            xlabel=r"$k \quad [\frac{1}{\rm{Mpc}}]$",
            ylabel=r"$\hat{\Theta}_0$",
            xscale="log",
            yscale="log",
        )
        ax.scatter(karr_ls, ls, label="ls", s=4)
        ax.scatter(karr_ss, ss, label="ss", s=5)
        ax.plot(x, y, ls=":", c="k")
        ax.legend()

        return fig


##############################################################################
# Hu and Sugiyama set of functions


@numba.njit
def _r_s(a: N, /, Req: float, keq: float) -> N:
    r"""Sound horizon in WKB approximation (Hu and Sugyiama [1]_, eqn. B-6).

    Parameters
    ----------
    a : scalar or array-like
        Scale factor (:math:`R / R_{eq}`).
    keq : float
        'k' at equality from Hu & Sugiyama eq. D-8 and B-6 [1]_.

    Returns
    -------
    scalar or array-like

    References
    ----------
    .. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic Microwave
           Background: an Analytic Approach. \apj, 444, 489.
    """
    out: N =  (
        (2.0 / 3.0)
        / keq
        * sqrt(6.0 / Req)
        * log((sqrt(1.0 + a * Req) + sqrt(Req * (1.0 + a))) / (1.0 + sqrt(Req)))
    )
    return out


@numba.njit
def _J0(k: N, /, Req: float, keq: float) -> N:
    r"""Hu and Sugyiama [1]_, eqn. D-2.

    Parameters
    ----------
    k : scalar or array-like
        Value in units of inverse Megaparsec.

    Returns
    -------
    scalar or array-like

    References
    ----------
    .. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic Microwave
           Background: an Analytic Approach. \apj, 444, 489.
    """
    out: N = 0.25 * sqrt(1.5) * (keq / k) * Req
    return out


@numba.njit
def _qfunc(k: N, /, h: float, Ob0: float, Om0: float) -> N:
    r"""From [1]_ and [2]_.

    Parameters
    ----------
    k : scalar or array-like
        Value in units of inverse Megaparsec.
    h : float
        Hubble parameter.
    Ob0 : float
        Omega baryons: density of baryonic matter in units of the critical
        density at z=0.
    Om0 : float
        Omega matter: density of matter in units of the critical density at z=0.

    Returns
    -------
    scalar or array-like
        Same as ``k``.

    References
    ----------
    .. [1] Peacock, J., & Dodds, S. (1994). Reconstructing the Linear Power
           Spectrum of Cosmological Mass Fluctuations. MNRAS, 267, 1020.
    .. [2] Bardeen, J., Bond, J., Kaiser, N., & Szalay, A. (1986).
           The Statistics of Peaks of Gaussian Random Fields. \apj, 304, 15.
    """
    out: N = k * exp(2 * Ob0) / (Om0 * h ** 2)
    return out


@numba.njit
def _Temp(k: N, h: float, Ob0: float, Om0: float) -> N:
    r"""Temperature [K], from Hu & Sugiyama eq. A-21 [1]_.

    Parameters
    ----------
    k : scalar or array-like
        Value in units of inverse Megaparsec.
    h : float
        Hubble parameter.
    Ob0 : float
        Omega baryons: density of baryonic matter in units of the critical
        density at z=0.
    Om0 : float
        Omega matter: density of matter in units of the critical density at z=0.

    Returns
    -------
    scalar or array-like
        Same as ``k``.

    References
    ----------
    .. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic
           Microwave Background: an Analytic Approach. \apj, 444, 489.
    """
    q: N = _qfunc(k, h=h, Ob0=Ob0, Om0=Om0)
    qpows: N = 1 + 3.89 * q + (14.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4
    out: N = (log(1 + 2.34 * q) / (2.34 * q)) / sqrt(sqrt(qpows))
    return out


@numba.njit
def _UG(a: N) -> N:
    r"""Hu and Sugyiama [1]_, eqn. A-6.

    Parameters
    ----------
    a : scalar or array-like
        The scale factor.

    Returns
    -------
    scalar or array-like
        Same as ``a``.

    References
    ----------
    .. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic Microwave
           Background: an Analytic Approach. \apj, 444, 489.
    """
    out: N = (9 * a ** 3 + 2 * a ** 2 - 8 * a - 16 + 16 * sqrt(1 + a)) / (9 * a * (1 + a))
    return out


@numba.njit
def _DeltaBarT(a: N) -> N:
    r"""Hu and Sugyiama [1]_, eqn. A-16.

    Parameters
    ----------
    a : scalar or array-like
        The scale factor.

    Returns
    -------
    scalar or array-like
        Same as ``a``.

    References
    ----------
    .. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic Microwave
           Background: an Analytic Approach. \apj, 444, 489.
    """
    out: N = (1 + 0.4 * fnu * (1 - 0.333 * a / (1 + a))) * _UG(a)
    return out


@numba.njit
def _N2Bar(a: N) -> N:
    r"""Hu and Sugyiama [1]_, eqn. A-12.

    Parameters
    ----------
    a : scalar or array-like
        The scale factor.

    Returns
    -------
    scalar or array-like
        Same as ``a``.

    References
    ----------
    .. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic Microwave
           Background: an Analytic Approach. \apj, 444, 489.
    """
    out: N = (
        -0.1 * (20 * a * 19) / (3 * a + 4) * _UG(a)
        - 8 / 3.0 * a / (3 * a + 4)
        + 8 / 9.0 * log(1 + 0.75 * a)
    )
    return out


@numba.njit
def _PhiBar(a: N, k: N, /, keq: float) -> N:
    r"""Hu and Sugyiama [1]_, eqn. A-17.

    Parameters
    ----------
    a : scalar or array-like
        Scale factor.
    k : scalar or array-like
        Value in units of inverse Megaparsec.
    keq : float
        'k' at equality from Hu & Sugiyama eq. D-8 and B-6 [1]_.

    Returns
    -------
    scalar or array-like
        Same as ``a`` and ``k``.

    References
    ----------
    .. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic Microwave
           Background: an Analytic Approach. \apj, 444, 489.
    """
    out: N =  0.75 * keq / k ** 2 * (1 + a) / a ** 2 * _DeltaBarT(a)
    return out


@numba.njit
def PsiBar(a: N, k: N, /, keq: float) -> N:
    r"""Hu and Sugyiama [1]_, eqn. A-17.

    Parameters
    ----------
    a : scalar or array-like
        Scale factor.
    k : scalar or array-like
        Value in units of inverse Megaparsec.
    keq : float
        'k' at equality from Hu & Sugiyama eq. D-8 and B-6 [1]_.

    Returns
    -------
    scalar or array-like
        Same as ``a`` and ``k``.

    References
    ----------
    .. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic Microwave
           Background: an Analytic Approach. \apj, 444, 489.
    """
    out: N = (
        -0.75
        * (keq / k) ** 2
        * (1 + a)
        / a ** 2
        * (_DeltaBarT(a) + 1.6 * fnu * _N2Bar(a) / (1 + a))
    )
    return out


@numba.njit
def _Phi(a: N, k: N, /, keq: float, h: float, Ob0: float, Om0: float) -> N:
    r"""Hu and Sugyiama [1]_, eqn. A-22a.

    Parameters
    ----------
    a : scalar or array-like
        Scale factor.
    k : scalar or array-like
        Value in units of inverse Megaparsec.
    keq : float
        'k' at equality from Hu & Sugiyama eq. D-8 and B-6 [1]_.
    h : float
        Hubble parameter.
    Ob0 : float
        Omega baryons: density of baryonic matter in units of the critical
        density at z=0.
    Om0 : float
        Omega matter: density of matter in units of the critical density at z=0.

    Returns
    -------
    scalar or array-like
        Same as ``a`` and ``k``.

    References
    ----------
    .. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic Microwave
           Background: an Analytic Approach. \apj, 444, 489.
    """
    temp: N = _Temp(k, h=h, Ob0=Ob0, Om0=Om0)
    out: N = _PhiBar(a, k, keq=keq) * (temp + (1 - temp) * exp(-alpha1 * (a * k / keq) ** beta))
    return out


@numba.njit
def _Psi(a: N, k: N, /, keq: float, h: float, Ob0: float, Om0: float) -> N:
    r"""Hu and Sugyiama [1]_, eqn. A-22b.

    Parameters
    ----------
    a : scalar or array-like
        Scale factor.
    k : scalar or array-like
        Value in units of inverse Megaparsec.
    keq : float
        'k' at equality from Hu & Sugiyama eq. D-8 and B-6 [1]_.
    h : float
        Hubble parameter.
    Ob0 : float
        Omega baryons: density of baryonic matter in units of the critical
        density at z=0.
    Om0 : float
        Omega matter: density of matter in units of the critical density at z=0.

    Returns
    -------
    scalar or array-like
        Same as ``a`` and ``k``.

    References
    ----------
    .. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic Microwave
           Background: an Analytic Approach. \apj, 444, 489.
    """
    temp: N = _Temp(k, h=h, Ob0=Ob0, Om0=Om0)
    out: N = PsiBar(a, k, keq=keq) * (temp + (1 - temp) * exp(-alpha2 * (a * k / keq) ** beta))
    return out


@numba.njit
def _PhiBar0(k: N, /, keq: float) -> N:
    r"""by substituting R = 0 into PhiBar, DeltaBarT, UG.

    Parameters
    ----------
    k : scalar or array-like
        Value in units of inverse Megaparsec.
    keq : float
        'k' at equality from Hu & Sugiyama eq. D-8 and B-6 [1]_.

    Returns
    -------
    scalar or array-like
        Same as ``k``.
    """
    out: N = (5 + 2 * fnu) * keq ** 2 / (6 * k ** 2)
    return out


@numba.njit
def _PsiBar0(k: N, /, keq: float) -> N:
    r"""Hu and Sugyiama [1]_, eqn. A-19.

    Parameters
    ----------
    k : scalar or array-like
        Value in units of inverse Megaparsec.
    keq : float
        'k' at equality from Hu & Sugiyama eq. D-8 and B-6 [1]_.

    Returns
    -------
    array-like
        Same as ``k``.

    References
    ----------
    .. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic Microwave
           Background: an Analytic Approach. \apj, 444, 489.
    """
    out: N = -0.86 * _PhiBar0(k, keq=keq)
    return out


@numba.njit
def _Theta00(k: N, /, keq: float) -> N:
    r"""Hu and Sugyiama [1]_, eqn. A-19.

    Parameters
    ----------
    k : scalar or array-like
        Value in units of inverse Megaparsec.
    keq : float
        'k' at equality from Hu & Sugiyama eq. D-8 and B-6 [1]_.

    Returns
    -------
    array-like
        Same as ``k``.

    References
    ----------
    .. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic Microwave
           Background: an Analytic Approach. \apj, 444, 489.
    """
    out: N = 0.43 * _PhiBar0(k, keq=keq)
    return out


@numba.njit  # TODO! confirm this equation. It's different
def _PhiG(a: N, k: N, /, Req: float, keq: float, h: float, Ob0: float, Om0: float) -> N:
    r"""Hu and Sugyiama [1]_, eqn. D-4.

    Parameters
    ----------
    a : scalar or array-like
        Scale factor.
    k : scalar or array-like
        Value in units of inverse Megaparsec.
    Req : float
        Scale factor at equality.
    keq : float
        'k' at equality from Hu & Sugiyama eq. D-8 and B-6 [1]_.
    h : float
        Hubble parameter.
    Ob0 : float
        Omega baryons: density of baryonic matter in units of the critical
        density at z=0.
    Om0 : float
        Omega matter: density of matter in units of the critical density at z=0.

    Returns
    -------
    scalar or array-like
        Same as ``a`` and ``k``.

    References
    ----------
    .. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic Microwave
           Background: an Analytic Approach. \apj, 444, 489.
    """
    factor = 3.0 / 32.0 * (keq / k) ** 2 * Req
    term0 = power(1 + a * Req, -1.25) * (1 + factor * (2 - Req) + (1 + factor) * a * Req)
    term1 = power(1 + a * Req, 0.75)
    phi = _Phi(a, k, keq=keq, h=h, Ob0=Ob0, Om0=Om0)
    psi = _Psi(a, k, keq=keq, h=h, Ob0=Ob0, Om0=Om0)
    out: N = term0 * phi - term1 * psi
    return out


@llc_integrand
def _Ifunc_integrand(
    ap: float, args: T.Tuple[float, float, float, float, float, float, float]
) -> float:
    r"""Hu and Sugyiama [1]_, eqn. D-3.

    Parameters
    ----------
    ap : float
        Scale factor.
    args : tuple[float, float, float, float, float, float, float]
        ``a, k, Req, keq, h, Ob0, Om0``

        - a : Scale factor.
        - k : Value in units of inverse Megaparsec.
        - Req : Scale factor at equality.
        - keq : 'k' at equality from Hu & Sugiyama eq. D-8 and B-6 [1]_.
        - h : Hubble parameter.
        - Ob0 : density of baryonic matter in units of the critical density at z=0.
        - Om0 : density of matter in units of the critical density at z=0.

    Returns
    -------
    float

    References
    ----------
    .. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic Microwave
           Background: an Analytic Approach. \apj, 444, 489.
    """
    a, k, Req, keq, h, Ob0, Om0 = args
    rs: float = _r_s(a, Req=Req, keq=keq)
    rps: float = _r_s(ap, Req=Req, keq=keq)
    phig: float = _PhiG(ap, k, Req=Req, keq=keq, h=h, Ob0=Ob0, Om0=Om0)
    out: float = phig / ((1 + ap) / Req) ** 0.5 * sin(k * (rs - rps))
    return out


@np.vectorize
def _Ifunc(
    a: float,
    k: float,
    /,
    Req: float,
    keq: float,
    h: float,
    Ob0: float,
    Om0: float,
    *,
    integ_kw: Mapping = {},
) -> N:
    r"""Hu and Sugyiama [1]_, eqn. D-3.

    Parameters
    ----------
    a : scalar
        Scale factor. If 'a' is an array, `numpy.vectorize` performs this
        calculation element-wise.
    k : scalar
        Value in units of inverse Megaparsec. If 'a' is an array,
        `numpy.vectorize` performs this calculation element-wise.
    Req : float
        Scale factor at equality.
    keq : float
        'k' at equality from Hu & Sugiyama eq. D-8 and B-6 [1]_.
    h : float
        Hubble parameter.
    Ob0 : float
        Omega baryons: density of baryonic matter in units of the critical
        density at z=0.
    Om0 : float
        Omega matter: density of matter in units of the critical density at z=0.

    Returns
    -------
    scalar or array-like
        Same as ``a`` and ``k``.

    References
    ----------
    .. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic Microwave
           Background: an Analytic Approach. \apj, 444, 489.
    """
    args: T.Tuple[float, float, float, float, float, float, float]
    args = (a, k, Req, keq, h, Ob0, Om0)
    out: N = sqrt(2 / 3 / Req) * integ.quad(_Ifunc_integrand, 0, a, args=args, **integ_kw)[0]
    return out


def _Theta0hatSS(
    a: N,
    k: N,
    /,
    Req: float,
    keq: float,
    h: float,
    Ob0: float,
    Om0: float,
    *,
    integ_kw: Mapping = {},
) -> N:
    r"""Hu and Sugyiama [1]_, eqn. D-1.

    Parameters
    ----------
    a : scalar or array-like
        Scale factor.
    k : scalar or array-like
        Value in units of inverse Megaparsec.
    Req : float
        Scale factor at equality.
    keq : float
        'k' at equality from Hu & Sugiyama eq. D-8 and B-6 [1]_.
    h : float
        Hubble parameter.
    Ob0 : float
        Omega baryons: density of baryonic matter in units of the critical
        density at z=0.
    Om0 : float
        Omega matter: density of matter in units of the critical density at z=0.

    Returns
    -------
    scalar or array-like
        Same as ``a`` and ``k``.

    References
    ----------
    .. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic Microwave
           Background: an Analytic Approach. \apj, 444, 489.
    """
    phi: N = _Phi(a, k, keq=keq, h=h, Ob0=Ob0, Om0=Om0)
    ival: N = _Ifunc(a, k, Req=Req, keq=keq, h=h, Ob0=Ob0, Om0=Om0, integ_kw=integ_kw)
    rs: N = _r_s(a, Req=Req, keq=keq)
    theta0: N = -phi + (
        ival
        + (cos(k * rs) + _J0(k, Req=Req, keq=keq) * sin(k * rs))
        * (_Theta00(k, keq=keq) + _PhiBar0(k, keq=keq))
    ) / power(1 + a * Req, 0.25)
    return theta0


@llc_integrand
def _Theta0hatLS_integrand(
    ap: float, args: T.Tuple[float, float, float, float, float, float, float]
) -> float:
    r"""Hu and Sugyiama [1]_, eqn. D-6.

    Parameters
    ----------
    ap : float
        Scale factor.
    args : tuple[float, float, float, float, float, float, float]
        ``a, k, Req, keq, h, Ob0, Om0``

        - a : Scale factor.
        - k : Value in units of inverse Megaparsec.
        - Req : Scale factor at equality.
        - keq : 'k' at equality from Hu & Sugiyama eq. D-8 and B-6 [1]_.
        - h : Hubble parameter.
        - Ob0 : density of baryonic matter in units of the critical density at z=0.
        - Om0 : density of matter in units of the critical density at z=0.

    Returns
    -------
    float

    References
    ----------
    .. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic Microwave
           Background: an Analytic Approach. \apj, 444, 489.
    """
    a, k, Req, keq, h, Ob0, Om0 = args
    rs: float = _r_s(a, Req=Req, keq=keq)
    rps: float = _r_s(ap, Req=Req, keq=keq)
    phi: float = _Phi(ap, k, keq=keq, h=h, Ob0=Ob0, Om0=Om0)
    psi: float = _Psi(ap, k, keq=keq, h=h, Ob0=Ob0, Om0=Om0)
    out: float = keq * Req * sqrt((1 + ap) / 2) * (phi - psi) * sin(k * (rs - rps))
    return out


@np.vectorize
def _Theta0hatLS(
    a: float,
    k: float,
    /,
    Req: float,
    keq: float,
    h: float,
    Ob0: float,
    Om0: float,
    *,
    integ_kw: Mapping = {},
) -> N:
    r"""Hu and Sugyiama [1]_, eqn. D-6.

    Parameters
    ----------
    a : scalar
        Scale factor. If 'a' is an array, `numpy.vectorize` performs this
        calculation element-wise.
    k : scalar
        Value in units of inverse Megaparsec. If 'k' is an array,
        `numpy.vectorize` performs this calculation element-wise.
    Req : float
        Scale factor at equality.
    keq : float
        'k' at equality from Hu & Sugiyama eq. D-8 and B-6 [1]_.
    h : float
        Hubble parameter.
    Ob0 : float
        Omega baryons: density of baryonic matter in units of the critical
        density at z=0.
    Om0 : float
        Omega matter: density of matter in units of the critical density at z=0.

    Returns
    -------
    scalar or array-like
        Same as ``a`` and ``k``.

    References
    ----------
    .. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic Microwave
           Background: an Analytic Approach. \apj, 444, 489.
    """
    args: T.Tuple[float, float, float, float, float, float, float]
    args = (a, k, Req, keq, h, Ob0, Om0)
    val = integ.quad(_Theta0hatLS_integrand, 0.0, a, args=args, **integ_kw)[0]
    theta0: N = (
        -_Phi(a, k, keq=keq, h=h, Ob0=Ob0, Om0=Om0)
        + (_Theta00(k, keq=keq) + _PhiBar0(k, keq=keq)) * cos(k * _r_s(a, Req=Req, keq=keq))
        + k / sqrt(3) * val
    )
    return theta0