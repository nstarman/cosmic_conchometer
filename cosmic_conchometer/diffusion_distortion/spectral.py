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
from astropy.cosmology.core import Cosmology
from classy import Class as CLASS
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.widgets import Button, Slider, TextBox
from numpy import absolute, arctan2, array, nan_to_num, sqrt
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.signal import find_peaks
from scipy.special import xlogy

# PROJECT-SPECIFIC
from .core import DiffusionDistortionBase
from cosmic_conchometer.typing import ArrayLikeCallable
from cosmic_conchometer.utils import distances

##############################################################################
# PARAMETERS

I = int
rho0 = distances.rho0

##############################################################################
# CODE
##############################################################################


def rho2_of_rho1(rho1, spll, sprp, maxrhovalid):
    """:math:`rho_2 = rho_1 - \sqrt{(s_{\|}+rho_1-rho_V)^2 + s_{\perp}^2}`

    TODO! move to utils

    Parameters
    ----------
    rho1 : float
    spll, sprp : float

    Returns
    -------
    float
    """
    return rho1 - sqrt((spll + rho1 - maxrhovalid) ** 2 + sprp ** 2)


# /def


##############################################################################


class SpectralDistortion(DiffusionDistortionBase):
    """Spectral Distortion.

    Parameters
    ----------
    cosmo : :class:`~astropy.cosmology.core.Cosmology`
        The cosmology
    class_cosmo : :class:`~classy.Class`
    AkFunc: Callable or str or None (optional, keyword-only)

    """

    def __init__(
        self,
        cosmo: Cosmology,
        class_cosmo: CLASS,
        *,
        AkFunc: T.Union[str, ArrayLikeCallable, None] = None,
        **kwargs,
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

    # /def

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

        Returns
        -------
        `~astropy.units.Quantity`
            units of [Mpc**2 * units[AkFunc]]

        Other Parameters
        ----------------
        real_AK : bool or None (optional)
            Whether to use the real (True) or imaginary (False) component of
            Ak or return the whole complex factor (None, default)

        """
        raise NotImplementedError("TODO!")

    # /def

    def _sprpP_boundary_terms(self, rho1: float, spll: float, sprp: float) -> float:
        """

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

        return U * s2V - sUp * sW

    # /def

    def _sprpP_integrand(self, rho1: float, spll: float, sprp: float, abs: bool = False) -> float:
        """

        Parameters
        ----------
        rho1 : float
        spll, sprp : float
        abs : bool
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

        return absolute(sUpp * sW) if abs else sUpp * sW

    # /def

    def _sprpP_integral(
        self,
        spll: float,
        sprp: float,
        bounds: T.Optional[T.Tuple[float, float]] = None,
        *,
        peakskw: T.Optional[T.Dict[str, T.Any]] = None,
        integkw: T.Optional[T.Dict[str, T.Any]] = None,
    ) -> T.Tuple[float, float, float, np.ndarray]:
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

    # /def

    def sprpP(
        self,
        spll: float,
        sprp: float,
        bounds: T.Optional[T.Tuple[float, float]] = None,
        *,
        peakskw: T.Optional[T.Dict[str, T.Any]] = None,
        integkw: T.Optional[T.Dict[str, T.Any]] = None,
    ):
        """

        Parameters
        ----------
        spll, sprp : float
        bounds : tuple[float, float] or None, optional
            The bounds of integration.
        **integkw
            For `scipy.integrate.quad`

        Returns
        -------
        s⟂P(s∥, s⟂) : float
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

    # /def

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
    # /def

    #######################################################
    # PLOT

    def plot_rho2(self, rho1=None, splls=None, sprps=None):
        rho1 = rho1 if rho1 is not None else self.rho_recombination
        splls = splls or np.linspace(0, 5.5, num=30)
        sprps = sprps or np.linspace(1e-4, 4, num=30)

        Splls, Sprps = np.meshgrid(splls, sprps)
        rhov = self.rhovalid[-1]

        fig, ax = plt.subplots(figsize=(7, 5))
        plt.subplots_adjust(bottom=0.25)  # !

        cnorm = Normalize(vmin=-4, vmax=1.5)
        scat = ax.scatter(Splls, Sprps, c=rho2_of_rho1(rho1, Splls, Sprps, rhov), s=30, norm=cnorm)
        ax.axvline(
            rhov - self.rho_recombination,
            c="k",
            label=r"$\rho_{valid}^{\max}-\rho_{R}$",
        )
        plt.colorbar(
            scat,
        )

        ax.set_title(r"$\rho_2(\rho_1 =" + f"{rho1:.3}" + r", s_{||}, s_{\perp}))$")
        ax.set_xlabel(r"$s_{||}$", fontsize=15)
        ax.set_ylabel(r"$s_{\perp}$", fontsize=15)
        ax.legend()

        # ----------------
        # interactivity

        axcolor = "lightgoldenrodyellow"
        axrho = plt.axes([0.12, 0.10, 0.65, 0.03], facecolor=axcolor)

        srho = Slider(axrho, r"$\rho_1$", self.class_rho.min(), self.class_rho.max(), valinit=rho1)

        def update(val):
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

        def submit(strval):
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

        def reset(event):
            srho.reset()

        button.on_clicked(reset)

        return fig

    # /def

    def plot_integrand_at(
        self,
        spll: float,
        sprp: float,
        *,
        integbounds: T.Optional[T.Tuple[float, float]] = None,
        peakskw: T.Optional[T.Dict[str, T.Any]] = None,
        integkw: T.Optional[T.Dict[str, T.Any]] = None,
        pdfmetadata: T.Optional[dict] = None,
    ):
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
            spll, sprp, bounds=integbounds, peakskw=peakskw, integkw=integkw
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

    # /deff


# /class

##############################################################################
# END
