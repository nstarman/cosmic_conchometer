# -*- coding: utf-8 -*-

"""Transfer Function from Hu and Sugiyama 1995 [1]_.

References
----------
.. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic Microwave
       Background: an Analytic Approach. \apj, 444, 489.
"""


##############################################################################
# IMPORTS

# STDLIB
import abc
import typing as T
from collections.abc import Mapping

# THIRD PARTY
import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy.integrate as integ
from astropy.cosmology.core import Cosmology
from numpy import cos, exp, log, power, sin, sqrt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import brentq

# PROJECT-SPECIFIC
from .base import TransferFunctionBase
from .utils import llc_integrand
from cosmic_conchometer.common import CosmologyDependent
from cosmic_conchometer.typing import ArrayLike as N

__all__ = ["TransferFunctionHuSugiyama1995Theta0"]

##############################################################################
# PARAMETERS

speed_of_light = const.c.to_value(u.km / u.s)

alpha1: float = 0.11
alpha2: float = 0.097
beta: float = 1.6
fnu: float = 0.45  # HS between A - 14 and A - 15 ratio of neutrino energy density to total radiation energy density


##############################################################################


class TransferFunctionHuSugiyama1995Theta0(TransferFunctionBase):
    """Transfer Function from Poor Man's Boltzmann Code in Hu and Sugiyama 1995.

    Parameters
    ----------
    cosmo : `~astropy.cosmology.core.Cosmology` instance
    a : float
        The scale factor.
    krange : |Quantity|
        The (lower, change, upper) bound in units of inverse Megaparsecs.
        'change' is the guessed 'k' to change between the two approximation
        regimes.
    lgkwindow : float
        The window width in log-space around k-'change' about which to search
        for k-cross, where the two approximations intersect. k-cross is the
        where to change between the two approximation regimes.
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

    _scale_factor: float
    _knum: int
    _kmin: u.Quantity
    _kchng: u.Quantity
    _kmax: u.Quantity
    _fit: np.polynomial.Polynomial
    _fit_info: T.Tuple

    def __init__(
        self,
        cosmo: Cosmology,
        a: float,
        As: float,
        *,
        krange: u.Quantity = (1e-4, 0.03, 10.0) / u.Mpc,
        lgkwindow: float = 0.05,
        knum: int = 100,
        domain: T.Optional[u.Quantity] = None,
        integ_kw: T.Optional[Mapping] = None,
        meta: T.Optional[Mapping] = None,
    ):
        super().__init__(cosmo=cosmo, meta=meta)

        self._integ_kw = integ_kw or {}

        # dimensionless radius  # TODO! recast as scale factor
        self._scale_factor = a
        if not np.isscalar(a):
            raise ValueError(f"`a` must be a scalar, not {type(a)}.")

        self._As = As

        # k
        self._kmin, self._kchng, self._kmax = krange.to(1 / u.Mpc)
        self._lgkwindow = lgkwindow
        self._knum = knum

        # evaluate lower approximation (overshoot kchange by lgkwindow)
        karr_ls: u.Quantity
        karr_ls = np.geomspace(self._kmin, self._kchng * 10 ** lgkwindow, knum, endpoint=False)
        _ls = self.Theta0hatLS(a, karr_ls, integ_kw=integ_kw)
        ls = T.cast(np.ndarray, _ls)  # TODO! fix type inference in Theta0hatLS

        # evaluate upper approximation (undershoot kchange by lgkwindow)
        karr_ss: u.Quantity
        karr_ss = np.geomspace(self._kchng / 10 ** lgkwindow, self._kmax, knum)
        _ss = self.Theta0hatSS(a, karr_ss, integ_kw=integ_kw)
        ss = T.cast(np.ndarray, _ss)  # TODO! fix type inference in Theta0hatSS

        self._kcross = self._find_kcross(karr_ls, ls, karr_ss, ss)

        lowk = karr_ls < self._kcross
        highk = karr_ss >= self._kcross
        # put the results together
        lgk = np.log10(np.concatenate((karr_ls.value[lowk], karr_ss.value[highk]))) << u.dex(
            1 / u.Mpc
        )
        th0 = np.concatenate((ls[lowk], ss[highk])) / self._As
        self._lgk = lgk
        self._th0 = th0

        # Find the first zero crossing (not in log space).
        zerox_idx = np.where(th0 <= 0)[0][0]
        lgk_zerox = lgk[zerox_idx]
        self._lgk_zerox = lgk_zerox

        # all the fits will be done on the absolute value of th0, so to be able
        # to recover the correct sign from the fit we need to store it now.
        # TODO? an actual step function
        self._signk = InterpolatedUnivariateSpline(lgk, np.sign(th0))

        # fit at low k
        # theta0 adjusted to factor out the dominant k dependence (k^2)
        lowk = slice(None, zerox_idx)
        th0adj = np.log10(np.abs(th0[lowk])) * lgk[lowk].value ** 2
        self._fitlowk = InterpolatedUnivariateSpline(lgk[lowk], th0adj[lowk])

        # fit at high k
        # theta0 adjusted to factor out the dominant k dependence (k^2)
        highk = slice(zerox_idx, None)
        th0adj2 = np.sign(th0[highk]) * np.power(np.abs(th0[highk]), lgk[highk].value ** 2)
        self._fithighk = InterpolatedUnivariateSpline(lgk[highk], th0adj2)

    def _find_kcross(
        self, klow: np.ndarray, th0low: np.ndarray, kup: np.ndarray, th0up: np.ndarray
    ) -> u.Quantity:
        # now spline them separately
        a = self._kchng / 10 ** self._lgkwindow
        b = self._kchng * 10 ** self._lgkwindow

        # low one
        sel_low = klow - a >= 0
        flowk = InterpolatedUnivariateSpline(klow[sel_low] << 1 / u.Mpc, th0low[sel_low])

        # high one
        sel_up = kup - b <= 0
        fhighk = InterpolatedUnivariateSpline(kup[sel_up] << 1 / u.Mpc, th0up[sel_up])

        g = lambda k: fhighk(k) - flowk(k)
        kcross = brentq(
            g,
            a.to_value(1 / u.Mpc),
            b.to_value(1 / u.Mpc),
            args=(),
            xtol=2e-12,
            rtol=8.881784197001252e-16,
            maxiter=100,
            full_output=False,
            disp=True,
        )

        kx = kcross / u.Mpc
        return kx

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
            k.to_value(1 / u.Mpc),  # * const.c.to_value(u.km / u.s),
            Req=self.Req,
            keq=self.keq.to_value(1 / u.Mpc),  # * const.c.to_value(u.km / u.s),
            h=self.cosmo.h,
            Ob0=self.cosmo.Ob0,
            Om0=self.cosmo.Om0,
            integ_kw=integ_kw or self._integ_kw,  # use default if integ_kw is None
        )
        return self._As * ss

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
            k.to_value(1 / u.Mpc),  # * const.c.to_value(u.km / u.s),
            Req=self.Req,
            keq=self.keq.to_value(1 / u.Mpc),  # * const.c.to_value(u.km / u.s),
            h=self.cosmo.h,
            Ob0=self.cosmo.Ob0,
            Om0=self.cosmo.Om0,
            integ_kw=integ_kw or self._integ_kw,  # use default if integ_kw is None
        )
        return self._As * ls

    def __call__(self, k: u.Quantity) -> N:
        """Evaluate :math:`\hat{\Theta}_0`, from Hu and Sugyiama [1]_.

        Parameters
        ----------
        k : |Quantity| ['wavenumber']
            In units of inverse Megaparsec.

        Returns
        -------
        float

        References
        ----------
        .. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic
               Microwave Background: an Analytic Approach. \apj, 444, 489.
        """
        lgks: np.ndarray = np.atleast_1d(np.log10(k.to_value(1 / u.Mpc)))

        abstheta0 = np.empty_like(lgks)

        lowk = lgks < self._lgk_zerox.value  # point where change fits
        abstheta0[lowk] = np.power(10, self._fitlowk(lgks[lowk]) / lgks[lowk] ** 2)

        highk = ~lowk
        abstheta0[highk] = np.power(np.abs(self._fithighk(lgks[highk])), 1 / lgks[highk] ** 2)

        theta0: N = self._signk(lgks) * abstheta0
        return theta0

    # ===============================================================
    # Plot

    def plot_theta0hat(self, fit: bool = True) -> plt.Figure:
        """Plot :math:`\hat{\Theta}_0(|k|; \rho)`.

        Parameters
        ----------
        fit : bool, optional
            Whether to plot the spline fit.
        """
        k = self._lgk.to(1 / u.Mpc)
        lowk = k < self._kcross

        # plot
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.add_subplot(
            111,
            xlabel=r"$k \quad [\frac{1}{\rm{Mpc}}]$",
            ylabel=r"$\hat{\Theta}_0 / A_s$",
            xscale="log",
            yscale="symlog",
        )
        ax.scatter(k[lowk], self._th0[lowk], label="ls", s=4)
        ax.scatter(k[~lowk], self._th0[~lowk], label="ss", s=4)
        if fit:
            ax.plot(k, self(k), ls=":", c="k")

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
    Req : float
        Scale factor at equality.
    keq : float
        'k' at equality from Hu & Sugiyama eq. D-8 and B-6 [1]_.

    Returns
    -------
    scalar or array-like ['Mpc']

    References
    ----------
    .. [1] Hu, W., & Sugiyama, N. (1995). Anisotropies in the Cosmic Microwave
           Background: an Analytic Approach. \apj, 444, 489.
    """
    out: N = (
        (2.0 / 3.0)
        / keq
        * sqrt(6.0 / Req)
        * log((sqrt(1.0 + a * Req) + sqrt(Req * (1.0 + a))) / (1.0 + sqrt(Req)))
    )
    return out


@numba.njit
def _J0(k: N, /, Req: float, keq: float) -> N:
    r"""Hu and Sugyiama [1]_, eqn. D-2, evaluated at eta=0.

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
def _transfer(k: N, h: float, Ob0: float, Om0: float) -> N:
    r"""Transfer function, from Hu & Sugiyama eq. A-21 [1]_.

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
        -0.1 * (20 * a + 19) / (3 * a + 4) * _UG(a)
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
    out: N = 0.75 * (keq / k) ** 2 * (1 + a) / a ** 2 * _DeltaBarT(a)
    return out


@numba.njit
def _PsiBar(a: N, k: N, /, keq: float) -> N:
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
    T: N = _transfer(k, h=h, Ob0=Ob0, Om0=Om0)
    out: N = _PhiBar(a, k, keq=keq) * (T + (1 - T) * exp(-alpha1 * (a * k / keq) ** beta))
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
    temp: N = _transfer(k, h=h, Ob0=Ob0, Om0=Om0)
    out: N = _PsiBar(a, k, keq=keq) * (temp + (1 - temp) * exp(-alpha2 * (a * k / keq) ** beta))
    return out


@numba.njit
def _PhiBar0(k: N, /, keq: float) -> N:
    r"""By substituting R = 0 into PhiBar, DeltaBarT, UG.

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
    # NOTE! have we correctly taken the limit!!!! THIS IS WORTH CHECKING
    out: N = (5 + 2 * fnu) * (keq / k) ** 2 / 6
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


@llc_integrand  # type: ignore  # diff btwn tuple[float, ...] & tuple[float, float]
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


@llc_integrand  # type: ignore  # diff btwn tuple[float, ...] & tuple[float, float]
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
