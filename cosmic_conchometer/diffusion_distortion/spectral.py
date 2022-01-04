# -*- coding: utf-8 -*-

"""Spectral Distortion."""

__all__ = ["SpectralDistortion"]


##############################################################################
# IMPORTS

# STDLIB
import typing as T

# THIRD PARTY
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integ
import tqdm
from astropy.cosmology.core import Cosmology
from classy import Class as CLASS
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.widgets import Button, Slider, TextBox
from numpy import absolute, arctan2, array, nan_to_num, sqrt
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.signal import find_peaks
from scipy.special import xlogy
from shapely.geometry import Point, Polygon

# PROJECT-SPECIFIC
from .core import DiffusionDistortionBase
from cosmic_conchometer.typing import ArrayLike, ArrayLikeCallable
from cosmic_conchometer.utils import distances

##############################################################################
# CODE
##############################################################################


def rho2_of_rho1(
    rho1: ArrayLike, spll: ArrayLike, sprp: ArrayLike, maxrhovalid: ArrayLike
) -> ArrayLike:
    """:math:`rho_2 = rho_1 - \sqrt{(s_{\|}+rho_1-rho_V)^2 + s_{\perp}^2}`

    TODO! move to utils

    Parameters
    ----------
    rho1 : float
    spll, sprp : float
    maxrhovalid : float

    Returns
    -------
    float
    """
    rho2: ArrayLike = rho1 - sqrt((spll + rho1 - maxrhovalid) ** 2 + sprp ** 2)
    return rho2


##############################################################################


class SpectralDistortion(DiffusionDistortionBase):
    """Spectral Distortion.

    Parameters
    ----------
    cosmo : :class:`~astropy.cosmology.core.Cosmology`
        The cosmology
    class_cosmo : :class:`~classy.Class`
    AkFunc: Callable or str or None (optional, keyword-only)
    **kwargs
        Ignored.
    """

    def __init__(
        self,
        cosmo: Cosmology,
        class_cosmo: CLASS,
        *,
        AkFunc: T.Union[str, ArrayLikeCallable, None] = None,
        **kwargs: T.Any,
    ) -> None:
        super().__init__(cosmo=cosmo, class_cosmo=class_cosmo, AkFunc=AkFunc, **kwargs)

        # ---------------------------------------
        # splines & related

        rho = self.class_rho

        self._spl_PbarCL = IUS(rho, self.class_P, ext=2)
        self._spl_dk_dt = IUS(rho, self.class_dk_dt, ext=2)
        self._spl_gbarCL = IUS(
            rho, self.class_g, ext=1
        )  # ext 1 makes stuff 0  # TODO! does this introduce a discontinuity?

        # derivatives wrt rho  # TODO! plot out to make sure they look good
        self._spl_d2k_dtdrho = self._spl_dk_dt.derivative(n=1)
        self._spl_d3k_dtdrho2 = self._spl_dk_dt.derivative(n=2)
        self._spl_d1gbarCL = self._spl_gbarCL.derivative(n=1)
        self._spl_d2gbarCL = self._spl_gbarCL.derivative(n=2)

    # ===============================================================

    @u.quantity_input(freq=u.GHz, k=1 / u.Mpc)
    def prefactor(
        self,
        freq: u.Quantity,
        k: u.Quantity,
        real_AK: T.Optional[bool] = None,
    ) -> u.Quantity:
        r"""Combined Prefactor.

        .. math::

            \frac{\lambda_0^2 A(\vec{k})}{16\pi \bar{P}_{\gamma}^{CL}(\zeta_0)}
            \frac{h\nu/k_B T_0}{e^{-\frac{h\nu}{k_B T_0}}-1}

        Parameters
        ----------
        freq : `~astropy.units.Quantity`
            The frequency.
        k : `~astropy.units.Quantity`
            Units of inverse Mpc.
        real_AK : bool or None (optional)
            Whether to use the real (`True`) or imaginary (`False`) component
            of ``Ak`` or return the whole complex factor (`None`, default).

        Returns
        -------
        `~astropy.units.Quantity`
            units of [Mpc**2 * units[AkFunc]]
        """
        raise NotImplementedError("TODO!")

    def _sprpP_boundary_terms(self, rho1: float, spll: float, sprp: float) -> float:
        """Boundary terms for s⟂P(s∥, s⟂) integration-by-parts.

        Parameters
        ----------
        rho1 : float
        spll, sprp : float

        Returns
        -------
        float
        """
        delta = spll + rho1 - self.maxrhovalid
        s2 = delta ** 2 + sprp ** 2
        s = sqrt(s2)
        rho2 = rho1 - s
        gbarCL = self._spl_gbarCL(rho2)
        dkdt = self._spl_dk_dt(rho1)

        U = dkdt * gbarCL
        s2V = -nan_to_num(sprp ** 2 * delta / s2) + 1.5 * sprp * arctan2(delta, sprp)

        sUp = (sprp * self._spl_d2k_dtdrho(rho1) * gbarCL) + (
            dkdt * self._spl_d1gbarCL(rho2) * (sprp - nan_to_num(sprp * delta / s))
        )
        sW = 1.5 * delta * arctan2(delta, sprp) - xlogy(sprp, s2)

        bound: float = U * s2V - sUp * sW
        return bound

    def _sprpP_integrand(
        self, rho1: ArrayLike, spll: ArrayLike, sprp: ArrayLike, abs: bool = False
    ) -> ArrayLike:
        """Integrand for s⟂P(s∥, s⟂) integration-by-parts.

        Parameters
        ----------
        rho1 : float
        spll, sprp : float
        abs : bool, optional
            Whether to take the absolute value

        Returns
        -------
        float
        """
        delta = spll + rho1 - self.maxrhovalid
        s2 = delta ** 2 + sprp ** 2
        s = sqrt(s2)
        rho2 = rho1 - s
        d1gbarCL_val = self._spl_d1gbarCL(rho2)

        sUpp = (
            sprp * self._spl_d3k_dtdrho2(rho1) * self._spl_gbarCL(rho2)
            + 2
            * self._spl_d2k_dtdrho(rho1)
            * d1gbarCL_val
            * (sprp - nan_to_num((sprp * delta) / s))
            + self._spl_dk_dt(rho1)
            * (
                self._spl_d2gbarCL(rho2) * (sqrt(sprp) - nan_to_num((sqrt(sprp) * delta) / s)) ** 2
                - d1gbarCL_val * nan_to_num(sprp / s) ** 3
            )
        )
        sW = 1.5 * delta * arctan2(delta, sprp) - xlogy(sprp, s2)

        integrand: float = absolute(sUpp * sW) if abs else sUpp * sW
        return integrand

    def _sprpP_integral(
        self,
        spll: float,
        sprp: float,
        bounds: T.Optional[T.Tuple[float, float]] = None,
        *,
        peakskw: T.Optional[T.Dict[str, T.Any]] = None,
        integkw: T.Optional[T.Dict[str, T.Any]] = None,
    ) -> T.Tuple[float, float, float, np.ndarray]:
        """s⟂P(s∥, s⟂) integration-by-parts.

        See :meth:`SpectralDistortion.sprpP`.

        Returns
        -------
        res : float
            The integration.
        err : float
            The absolute error in the integral.
        rel_err : float
            The relative error in the integral.
        peaks : ndarray[float]
            Locations (in rho) of the peaks used to subdivide the integration
            bounds into well-behaved regions.
        """
        bounds = self.rhovalid if bounds is None else bounds

        x = self.class_rho[(bounds[0] <= self.class_rho) & (self.class_rho <= bounds[1])]
        vals = self._sprpP_integrand(x, spll, sprp, abs=True)
        pospeaks, posprops = find_peaks(vals, **(peakskw or {}))
        negpeaks, negprops = find_peaks(-vals, **(peakskw or {}))

        peaks = np.unique(
            np.concatenate(
                (
                    [bounds[0]],
                    self.class_rho[pospeaks],
                    self.class_rho[negpeaks],
                    [bounds[1]],
                )
            )
        )
        peaks.sort()  # in-place

        # TODO! identify which intervals are complaining
        ress, _ = np.array(
            [
                integ.quad(
                    self._sprpP_integrand,
                    lb,
                    ub,
                    args=(spll, sprp, False),
                    full_output=False,
                    **(integkw or {}),
                )
                for (lb, ub) in zip(peaks[:-1], peaks[1:])
            ]
        ).T

        absress, errs = np.array(
            [
                integ.quad(
                    self._sprpP_integrand,
                    lb,
                    ub,
                    args=(spll, sprp, True),
                    full_output=False,
                    **(integkw or {}),
                )
                for (lb, ub) in zip(peaks[:-1], peaks[1:])
            ]
        ).T

        res = ress.sum()
        err = np.sqrt(np.sum(np.square(errs)))
        rel_err = nan_to_num(err / absress.sum(), nan=0)

        return res, err, rel_err, peaks

    def sprpP(
        self,
        spll: float,
        sprp: float,
        bounds: T.Optional[T.Tuple[float, float]] = None,
        *,
        peakskw: T.Optional[T.Dict[str, T.Any]] = None,
        integkw: T.Optional[T.Dict[str, T.Any]] = None,
    ) -> T.Tuple[u.Quantity, u.Quantity, float]:
        r""":math:`s_{\perp}\mathcal{P}(s_{||},s_{\perp})`.

        Parameters
        ----------
        spll, sprp : float
        bounds : tuple[float, float] or None, optional
            The bounds of integration. If `None` (default) uses :attr:`~SpectralDistortion.rhovalid`
        peaksks : dict[str, Any] or None (optional, keyword-only)
            Keyword arguments for `scipy.signal.find_peaks`.
        integkw : dict[str, Any] or None (optional, keyword-only)
            Keyword arguments for `scipy.integrate.quad`.

        Returns
        -------
        s⟂P(s∥, s⟂) : float
        err : float
            The absolute error.
        rel_err : float
            The relative error.
        """
        prefactor = 3 * self.lambda0 ** 2 / (8 * self._spl_PbarCL(self.maxrhovalid))

        # boundary terms
        bounds = self.rhovalid if bounds is None else bounds
        ub = self._sprpP_boundary_terms(bounds[-1], spll, sprp)
        lb = self._sprpP_boundary_terms(bounds[0], spll, sprp)

        integral, err, rel_err, _ = self._sprpP_integral(
            spll, sprp, bounds=bounds, peakskw=peakskw, integkw=integkw
        )

        return prefactor * (ub - lb + integral), prefactor * err, rel_err

    @staticmethod
    def make_secondlast_scattering_integration_domain(Splls, Sprps, sprpP, threshold=1e-4, buffer=0.2):
        """Get the set of points in spll, sprp s.t. results > threshold.

        Parameters
        ----------
        Splls : (N, M) ndarray
            The spll coordinates as a meshgrid of shape (spll, sprp)
        Sprps : (N, M) ndarray
            The sprp coordinates as a meshgrid of shape (spll, sprp)
        results : (N, M) ndarray
            :math:`s_{\perp} P(s_{||}, s_{\perp})` over the meshgrid
        threshold : float, optional
            The minimum value in ``results`` to be kept.
            This defines the unbuffered shape.
        buffer : float
            The distance about the threshold-shape to buffer.
            This is in linear spll, sprp coordinates with a Euclidean metric.

        Returns
        -------
        `shapely.geometry.Polygon`
        """
        from shapely.geometry import Polygon
        
        lefts, rights = [], []
        for j, sprp in enumerate(Sprps[0, :]):
            where = np.where(sprpP[:, j] > threshold)[0]
            if not np.any(where):
                continue
            imin, imax = where[[0, -1]]
            lefts.append((Splls[imin, j], sprp))
            rights.append((Splls[imax, j], sprp))
        
        lefts = np.array(lefts)
        rights = np.array(rights)
        
        exterior = np.concatenate((lefts, rights[::-1]))
        
        poly = Polygon(shell=exterior)
        domain = poly.buffer(buffer)
        
        return domain

    def sprpP_on_grid(
        self,
        splls: np.ndarray,
        sprps: np.ndarray,
        bounds: T.Optional[T.Tuple[float, float]] = None,
        *,
        peakskw: T.Dict[str, T.Any] = dict(prominence=1e-7),
        integkw: T.Dict[str, T.Any] = dict(epsabs=1e-11, limit=500),
        domain: T.Optional[Polygon] = None,
    ) -> None:
        """Calculate  :math:`s_{\perp}\mathcal{P}(s_{||},s_{\perp})`.

        The calculated result is saved to an npz file.

        Parameters
        ----------
        splls : (N,) ndarray
            Array of :math:`s_{||}`. Built into a meshgrid with ``sprps``.
        sprps : (M,) ndarray
            Array of :math:`s_{\perp}`. Built into a meshgrid with ``splls``.
        bounds : tuple[float, float] or None, optional
            The bounds of integration. If `None` (default) uses
            :attr:`~SpectralDistortion.rhovalid`, buffering the upper bound by 3.
        """
        # make meshgrid
        Sprps, Splls = np.meshgrid(sprps, splls)

        # Define bounds of integration. Build default, if None.
        bounds = (self.rhovalid[0], self.maxrhovalid + 3) if bounds is None else bounds
        bounds = T.cast(T.Tuple[float, float], bounds)

        num: int = len(Splls.flat)
        results: np.ndarray = np.full(num, np.NaN, dtype=float)
        errors: np.ndarray = np.full(num, np.NaN, dtype=float)
        relative_errors: np.ndarray = np.full(num, np.NaN, dtype=float)

        i: int
        spll: float
        sprp: float
        for i, (spll, sprp) in tqdm.tqdm(enumerate(zip(Splls.flat, Sprps.flat)), total=num):

            # check if the point is within the mask
            if domain is not None and not domain.contains(Point(spll, sprp)):
                continue

            # TODO! test upper bound doesn't matter so much, so long as upper > rhoV
            res, err, rel_err = self.sprpP(
                spll, sprp, bounds=bounds, peakskw=peakskw, integkw=integkw
            )

            results[i] = res.value
            errors[i] = err.value
            relative_errors[i] = rel_err

        # save mesh, sprpP integral, associated errors, and other info.
        np.savez(
            f"output/run-bounds_{bounds}-integkw_{integkw.items()}.npz",
            Splls=Splls,
            Sprps=Sprps,
            results=results,
            errors=errors,
            relative_errors=relative_errors,
            bounds=np.array(list(bounds)),
        )

    def fit_smooth_sP(self):
        pass

    # ===============================================================
    # Convenience Methods

    @u.quantity_input(freq=u.GHz, k=1 / u.Mpc, theta_kS=u.rad)
    def __call__(
        self,
        freq: u.Quantity,
        k: u.Quantity,
        theta_kS: u.Quantity,
        *,
        zeta_min: float = 0.0,
        zeta_max: float = np.inf,
        m_max: int = 100,
    ) -> u.Quantity:
        r"""Perform Calculation.

        The cross-terms in :math:`A(k)` with the integral cancel out.
        Only the real-with-real and imaginary-with-imaginary remain.

        Parameters
        ----------
        freq : `~astropy.units.Quantity`
            Frequency [GHz].
        k : `~astropy.units.Quantity`
            :math:`k` magnitude, [1/Mpc].
        theta_kS : `~astropy.units.Quantity`
            Angle between :math:`k` and scattering location, [radians].

        Returns
        -------
        spectral_distortion : `~astropy.units.Quantity`

        Other Parameters
        ----------------
        zeta_min, zeta_max : float (optional, keyword-only)
            Minimum / Maximum :math:`\zeta`
        m_max : int (optional, keyword-only)
            Maximum index in summation.

        """
        raise NotImplementedError("TODO!")

    compute = __call__

    #######################################################
    # PLOT

    def plot_rho2(
        self, rho1: float = None, splls: np.ndarray = None, sprps: np.ndarray = None
    ) -> plt.Figure:
        """Plot :math:`\rho_2(s_{||}, s_{\perp}; \rho_1)`.

        Parameters
        ----------
        rho1 : float, optional
            The rho of last sctter. If `None` (default) uses
            :attr:`SpectralDistortion.rho_recombination`.
        splls : (N,) ndarray or None, optional
            Array of :math:`s_{||}`. Built into a meshgrid with ``sprps``.
            If `None` (default), uses ``np.linspace(0, 5.5, num=30)``.
        sprps : (M,) ndarray, optional
            Array of :math:`s_{\perp}`. Built into a meshgrid with ``splls``.
            If `None` (default), uses ``np.linspace(1e-4, 4, num=30)``.

        Returns
        -------
        `~matplotlib.pyplot.Figure`
        """
        # get arguments, replacing with defaults if None.
        rho1 = rho1 if rho1 is not None else self.rho_recombination
        splls = splls or np.linspace(0, 5.5, num=30)
        sprps = sprps or np.linspace(1e-4, 4, num=30)

        # build meshgrid and anchor rho
        Splls, Sprps = np.meshgrid(splls, sprps)
        rhov = self.rhovalid[-1]

        # Plot
        fig, ax = plt.subplots(figsize=(7, 5))
        plt.subplots_adjust(bottom=0.25)

        cnorm = Normalize(vmin=-4, vmax=1.5)
        scat = ax.scatter(Splls, Sprps, c=rho2_of_rho1(rho1, Splls, Sprps, rhov), s=30, norm=cnorm)
        ax.axvline(rhov - self.rho_recombination, c="k", label=r"$\rho_{valid}^{\max}-\rho_{R}$")
        plt.colorbar(scat)

        ax.set_title(r"$\rho_2(\rho_1 =" + f"{rho1:.3}" + r", s_{||}, s_{\perp}))$")
        ax.set_xlabel(r"$s_{||}$", fontsize=15)
        ax.set_ylabel(r"$s_{\perp}$", fontsize=15)
        ax.legend()

        # ----------------
        # interactivity

        axcolor = "lightgoldenrodyellow"
        axrho = plt.axes([0.12, 0.10, 0.65, 0.03], facecolor=axcolor)

        srho = Slider(axrho, r"$\rho_1$", self.class_rho.min(), self.class_rho.max(), valinit=rho1)

        def update(val: T.Any) -> None:
            rho1 = srho.val
            scat = ax.scatter(
                Splls, Sprps, c=rho2_of_rho1(rho1, Splls, Sprps, rhov), s=30, norm=cnorm
            )
            ax.axvline(
                rhov - self.rho_recombination,
                c="k",
                label=r"$\rho_{valid}^{\max}-\rho_{R}$",
            )
            ax.set_title(r"$\rho_2(\rho_1 =" + f"{rho1:.3}" + r", s_{||}, s_{\perp}))$")
            # scat.set_facecolor(rho2_of_rho1(rho1, Splls, Sprps, rhov))
            # fig.axes[0].set_title(r"$\rho_2(\rho_1 =" + f"{rho1:.3}" + r", s_{||}, s_{\perp}))$")
            fig.canvas.draw_idle()

        srho.on_changed(update)

        axbox = fig.add_axes([0.12, 0.025, 0.2, 0.04])
        text_box = TextBox(axbox, r"$\rho_1$")

        def submit(strval: str) -> None:
            rho1 = float(strval)
            scat = ax.scatter(
                Splls, Sprps, c=rho2_of_rho1(rho1, Splls, Sprps, rhov), s=30, norm=cnorm
            )
            ax.axvline(
                rhov - self.rho_recombination,
                c="k",
                label=r"$\rho_{valid}^{\max}-\rho_{R}$",
            )
            ax.set_title(r"$\rho_2(\rho_1 =" + f"{rho1:.3}" + r", s_{||}, s_{\perp}))$")
            # scat.set_facecolor(rho2_of_rho1(rho1, Splls, Sprps, rhov))
            # fig.axes[0].set_title(r"$\rho_2(\rho_1 =" + f"{rho1:.3}" + r", s_{||}, s_{\perp}))$")
            fig.canvas.draw_idle()

        text_box.on_submit(submit)

        resetax = plt.axes([0.7, 0.025, 0.1, 0.04])
        button = Button(resetax, "Reset", color=axcolor, hovercolor="0.975")

        def reset(event: T.Any) -> None:
            srho.reset()

        button.on_clicked(reset)

        return fig

    def plot_integrand_at(
        self,
        spll: float,
        sprp: float,
        *,
        bounds: T.Optional[T.Tuple[float, float]] = None,
        peakskw: T.Optional[T.Dict[str, T.Any]] = None,
        integkw: T.Optional[T.Dict[str, T.Any]] = None,
        pdfmetadata: T.Optional[T.Dict[str, T.Any]] = None,
    ) -> plt.Figure:
        """Plot s⟂P(s∥, s⟂) integrand.

        Parameters
        ----------
        spll, sprp : float
        bounds : tuple[float, float] or None (optional, keyword-only)
            The bounds of integration. If `None` (default) uses :attr:`~SpectralDistortion.rhovalid`
        peaksks : dict[str, Any] or None (optional, keyword-only)
            Keyword arguments for `scipy.signal.find_peaks`.
        integkw : dict[str, Any] or None (optional, keyword-only)
            Keyword arguments for `scipy.integrate.quad`.
        pdfmetadata : dict[str, Any] or None (optional, keyword-only)
            Metadata for saved PDF file.

        Returns
        -------
        `~matplotlib.pyplot.Figure`
        """
        # create figure and axes
        fig = plt.figure(figsize=(12, 4))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 3, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharey=ax1)
        ax3 = fig.add_subplot(gs[2], sharey=ax2)

        # format ticks
        plt.setp(ax1.get_yticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)
        ax1.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax3.spines["left"].set_visible(False)
        ax1.yaxis.tick_left()
        ax3.yaxis.tick_right()

        # axes lines
        d = 0.015  # how big to make the diagonal lines in axes coordinates
        kw = dict(transform=ax1.transAxes, color="k", clip_on=False)
        ax1.plot((1 - d, 1 + d), (-d, +d), **kw)
        ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)

        kw.update(transform=ax2.transAxes)
        ax2.plot((-d, +d), (1 - d, 1 + d), **kw)
        ax2.plot((-d, +d), (-d, +d), **kw)
        ax2.plot((1 - d, 1 + d), (-d, +d), **kw)
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)

        kw.update(transform=ax3.transAxes)
        ax3.plot((-d, +d), (1 - d, 1 + d), **kw)
        ax3.plot((-d, +d), (-d, +d), **kw)

        # evaluate integrand
        vals = self._sprpP_integrand(self.class_rho, spll, sprp)
        res, err, rel_err, peaks = self._sprpP_integral(
            spll, sprp, bounds=bounds, peakskw=peakskw, integkw=integkw
        )

        # left-bound plot
        ax1.plot(self.class_rho, vals)
        ax1.axvline(peaks[0], color="k", alpha=0.7, ls=":", label="left bound")
        ax1.set_xlim(peaks[0] - 0.1, peaks[0] + 0.3)
        ax1.legend()

        # interesting plot
        ax2.plot(self.class_rho, vals)
        ax2.axvline(peaks[1], color="k", alpha=0.7, ls=":", label="extrama(|f(x)|)")
        [ax2.axvline(peak, color="k", alpha=0.7, ls=":") for peak in peaks[2:-1]]
        ax2.set_xlim(peaks[1] - 0.1, peaks[-2] + 0.1)
        ax2.legend()

        # right-bound plot
        ax3.plot(self.class_rho, vals)
        ax3.axvline(peaks[-1], color="k", alpha=0.7, ls=":", label="right bound")
        ax3.set_xlim(peaks[-1] - 0.3, peaks[-1] + 0.1)
        ax3.legend()

        return fig


##############################################################################
# END
