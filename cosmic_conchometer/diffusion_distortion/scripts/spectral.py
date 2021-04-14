# -*- coding: utf-8 -*-

"""**DOCSTRING**.

This script can be run from the command line with the following parameters:

Parameters
----------

"""

__all__ = [
    "make_parser",
    "main",
    # functions
    "scriptCnogam_component",
    "scriptCgamma_component",
]


##############################################################################
# IMPORTS

# BUILT-IN
import argparse
import itertools
import pathlib
import typing as T
import warnings

# THIRD PARTY
import numpy as np
from mpmath import expj, gamma, hyp1f2, hyp2f2, mpc, mpf, pi, power, sqrt
from scipy.special import spherical_jn as besselJ

# PROJECT-SPECIFIC
from cosmic_conchometer.data import DATA_DIR
from cosmic_conchometer.setup_package import HAS_TQDM, _NoOpPBar

if HAS_TQDM:
    # THIRD PARTY
    from tqdm import tqdm

##############################################################################
# PARAMETERS

sqrtpi = mpf(sqrt(pi))

# General
_PLOT: bool = True  # Whether to plot the output
_VERBOSE: bool = True  # Degree of logfile verbosity

# Specific
_BDMIN: float = 0.0
_BDMAX: float = 200.0
_BDSTEP: float = 1.0

_BIGMMAX: int = 100
_LITTLEMMAX: int = 30
_LMAX: int = 50  # 49 + 1

##############################################################################
# CODE
##############################################################################


@np.vectorize
def scriptCnogam_component(betaDelta: mpf, M: int, m: int, rhoES: int) -> mpc:
    r"""Eqn 109 of https://www.overleaf.com/project/5efe491b4140390001b1c892.

    .. math::

        (l+1)^M (\beta\Delta)^m {\cal C}(\beta\Delta,0,M,m;l) =
        \frac{1}{(\beta\Delta)} \left[
            \tilde{\rho}_{ES}^{M} ~ j_{m+1}(\beta\Delta \tilde{\rho}_{ES})
        \right]
        -
        \frac{\sqrt{\pi}\left(M-(2m^2+6m+5)\right)}
             {4(M+m+1)\Gamma\left(\frac{2m+5}{2}\right)}
        \left(\frac{\beta\Delta}{2}\right)^{-M-1}
        \Bigg[
            \left(\frac{l\beta\Delta}{2}\right)^{M+m+1}  \,
            {_1F_2}\left( \frac{M+m+1}{2};\frac{2m+5}{2},\frac{M+m+3}{2};
                          -\left(\frac{l \beta\Delta}{2}\right)^2        \right)
        \Bigg]

    Parameters
    ----------
    betaDelta : `~mpmath.mpf`
        .. math::

            \beta = |\vec{k}| \lambda_0 \cos{\theta_{kS}}
            \Delta = \frac{1}{N} \sqrt(\frac{(1+1/a_{eq})}{2})

    M, m : int
        arguments.

        .. todo:: rename m vs M. too confusing

    rhoES : int

    Returns
    -------
    `~mpmath.mpc`

    """
    x: mpf = betaDelta * rhoES

    t1: mpc = besselJ(m + 1, x) / power(betaDelta, m + 1)
    t2: mpc = (
        sqrtpi
        * (M - (2 * m ** 2 + 6 * m + 5))
        / (4 * (M + m + 1) * gamma(m + 2.5) * power(2, m))
    )
    t3: mpc = power(rhoES, m + 1) * hyp1f2(
        (M + m + 1) / 2,
        m + 2.5,
        (M + m + 3) / 2,
        -((rhoES * betaDelta / 2) ** 2),
    )

    c: mpc = t1 - t2 * t3
    return c


# /def


@np.vectorize
def scriptCgamma_component(betaDelta: mpf, M: int, m: int, rhoES: int) -> mpc:
    r"""Eqn 115 of https://www.overleaf.com/project/5efe491b4140390001b1c892

    .. math::

        l^M (\beta\Delta)^m \mathcal{C}_{component}(\beta\Delta,\beta\Delta,M,m;l) =
        \frac{1}{\beta\Delta} \Bigg[
            l^M \Exp{i\beta\Delta l}
            j_{m+1}(\beta\Delta  l)\
        \Bigg]
        - \frac{\sqrt{\pi} l^{M+1}(\beta\Delta  l)^m}
               {2^{m+2}
           \Gamma \left(\frac{2m\!+\!5}{2}\right)}
        \Bigg[
            \frac{i\beta\Delta  l}{(M\!+\!m\!+\!2)} \,
            {_2F_2}(m\!+\!2,m\!+\!M\!+\!2;2m\!+\!4,m\!+\!M\!+\!3;
                    2i \beta\Delta  l)
            - \frac{2m^2\!+\!6m\!-\!M\!+\!5}{(M\!+\!m\!+\!1)} \,
            {_2F_2}(m\!+\!2,m\!+\!M\!+\!1;2m\!+\!4,m\!+\!M\!+\!2;
                    2i \beta\Delta  l)
        \Bigg]

    Parameters
    ----------
    betaDelta : `~mpmath.mpf`
        .. math::

            \beta = |\vec{k}| \lambda_0 \cos{\theta_{kS}}
            \Delta = \frac{1}{N} \sqrt(\frac{(1+1/a_{eq})}{2})

    M, m : int
        arguments.

        .. todo:: rename m vs M. too confusing

    rhoES : int

    Returns
    -------
    `~mpmath.mpc`

    """
    x: mpf = betaDelta * rhoES

    t1: mpc = expj(x) * besselJ(m + 1, x) / power(betaDelta, m + 1)
    t2: mpc = sqrtpi / power(2, m + 2) * power(rhoES, m + 1) / gamma(m + 2.5)
    t3: mpc = (
        (1j * x)
        / (M + m + 2)
        * hyp2f2(m + 2, m + M + 2, 2 * m + 4, m + M + 3, 2j * x)
    )
    # (2m^2+6m-M+5) / (M+m+1) * hyp2f2(m+2, m+M+1, 2m+4, m+M+2, 2i x)
    t4: mpc = (
        (2 * m ** 2 + 6 * m - M + 5)
        / (M + m + 1)
        * hyp2f2(m + 2, M + m + 1, 2 * m + 4, m + M + 2, 2j * x)
    )

    c: mpc = t1 - t2 * (t3 - t4)
    return c


# /def

##############################################################################
# Command Line
##############################################################################


# TODO! rename m vs M. too confusing
def make_parser(
    *,
    BDmin: float = _BDMIN,
    BDmax: float = _BDMAX,
    BDstep: float = _BDSTEP,
    Mmax: int = _BIGMMAX,
    mmax: int = _LITTLEMMAX,
    lmax: int = _LMAX,
    # gamma type
    gamma: str = "both",
    # general
    data_dir: str = DATA_DIR,
    # plot: bool = _PLOT,
    verbose: bool = _VERBOSE,
    inheritable: bool = False,
) -> argparse.ArgumentParser:
    r"""Expose `~argparse.ArgumentParser` for :func:`~.main`.

    Parameters
    ----------
    BDmin, BDmax, BDstep : float (optional, keyword-only)
        :math:`\beta\Delta` minimum, maximum, and step, respectively for
        :func:`~numpy.arange`.

    Mmax, mmax, lmax : int (optional, keyword-only)
        The maximum value for index arguments M, m, l, respectively.
        All have minimum value 0.

    Returns
    -------
    parser:  `~argparse.ArgumentParser`
        The parser with arguments:

        - BDmin, BDmax, BDstep
        - Mmax, mmax, lmax
        - data_dir
        - verbose

    Other Parameters
    ----------------
    data_dir : str (optional, keyword-only)
        Where to save all the files.

    verbose : bool (optional, keyword-only)
        Script logging verbosity.

    inheritable: bool (optional, keyword-only)
        whether the parser can be inherited from (default False).
        if True, sets ``add_help=False`` and ``conflict_hander='resolve'``

    """
    parser = argparse.ArgumentParser(
        description="",
        add_help=not inheritable,
        conflict_handler="resolve" if not inheritable else "error",
    )

    parser.add_argument("--BDmin", action="store", default=BDmin, type=float)
    parser.add_argument("--BDmax", action="store", default=BDmax, type=float)
    parser.add_argument("--BDstep", action="store", default=BDstep, type=float)

    parser.add_argument("--Mmax", action="store", default=Mmax, type=int)
    parser.add_argument("--mmax", action="store", default=mmax, type=int)
    parser.add_argument("--lmax", action="store", default=lmax, type=int)

    parser.add_argument(
        "--gamma", choices=["gamma", "nogam", "both"], type=str, default="both"
    )

    # where to save stuff
    parser.add_argument(
        "--data_dir",
        action="store",
        default=data_dir,
        type=str,
    )
    # script verbosity
    parser.add_argument(
        "-v",
        "--verbose",
        action="store",
        default=verbose,
        type=bool,
    )

    # ------------------

    # Schwimmbad multiprocessing
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ncores",
        dest="n_cores",
        default=1,
        type=int,
        help="Number of processes (uses multiprocessing).",
    )
    group.add_argument(
        "--mpi",
        dest="mpi",
        default=False,
        action="store_true",
        help="Run with MPI.",
    )

    return parser


# /def


# ------------------------------------------------------------------------


class Worker:
    """Worker."""

    def __init__(self, opts: T.Mapping) -> None:
        # this defines the cube for all terms in the sums to perform the diffusion
        # distortion integral.
        L, m, M = np.mgrid[0 : opts.lmax, 0 : opts.mmax, 0 : opts.Mmax]
        # TODO! rename m vs M. too confusing

        self.L = L
        self.m = m
        self.M = M
        self.verbose = opts.verbose
        drct = pathlib.Path(opts.data_dir)

        # No-Gamma functions.
        nogam_folder = "scriptC_nogam"
        nogam_drct = drct.joinpath(nogam_folder)
        nogam_drct.mkdir(exist_ok=True)
        self.nogam_drct = nogam_drct

        # Gamma functions.
        gamma_folder = "scriptC_gamma"
        gamma_drct = drct.joinpath(gamma_folder)
        gamma_drct.mkdir(exist_ok=True)
        self.gamma_drct = gamma_drct

    # /def

    def compute_and_save(self, bD: mpf, gamma: str) -> mpc:
        """Compute and save.

        Parameters
        ----------
        bD : float
        gamma : bool
            Whether to calculate the gamma or no-gamma versions.

        Returns
        -------
        C : mpc

        """
        L, m, M = self.L, self.m, self.M
        bDstr = str(bD).replace(".", "_")

        # compute C
        if gamma:
            C = scriptCgamma_component(mpf(bD), M=M, m=m, rhoES=L)
            # save result
            np.save(self.gamma_drct.joinpath("scriptCgamma_comp-" + bDstr), C)

        else:
            C = scriptCnogam_component(mpf(bD), M=M, m=m, rhoES=L)
            # save result
            np.save(self.nogam_drct.joinpath("scriptCnogam_comp-" + bDstr), C)

        return C

    # /def

    def __call__(self, task: T.Tuple[mpf, str]) -> mpc:
        bD, gamma = task
        return self.compute_and_save(bD, gamma)

    # /def


# /class

# ------------------------------------------------------------------------


def main(
    args: T.Union[list, str, None] = None,
    opts: T.Optional[argparse.Namespace] = None,
) -> None:
    """Script Function.

    .. todo::

    Parameters
    ----------
    args : list or str or None (optional)
        an optional single argument that holds the sys.argv list,
        except for the script name (e.g., argv[1:])
    opts : `~argparse.Namespace` or None (optional)
        pre-constructed results of parsed args
        if not None, used ONLY if args is None

    """
    import schwimmbad

    p: argparse.Namespace
    if opts is not None and args is None:
        p = opts
    else:
        if opts is not None:
            warnings.warn("Not using `opts` because `args` are given")
        if isinstance(args, str):
            args = args.split()

        parser = make_parser()
        p = parser.parse_args(args)

    # /if

    # make the range of $\beta \Delta$ values over which to evaluate.
    # this is the independent and continuous variable and the result will
    # need to be interpolated as a function of $\beta \Delta$.
    # TODO! does this need to be mpf or mp.matrix?
    betaDeltas = np.arange(p.BDmin, p.BDmax, p.BDstep)

    if p.gamma != "both":
        iterator = ((bD, p.gamma) for bD in betaDeltas)
        leniter = len(betaDeltas)
    else:
        iterator = ((bD, g) for bD in betaDeltas for g in (False, True))
        leniter = 2 * len(betaDeltas)
        
    t = (
        tqdm(total=leniter)
        if (p.verbose and HAS_TQDM)
        else _NoOpPBar()
    )

    worker = Worker(p)
    pool = schwimmbad.choose_pool(mpi=p.mpi, processes=p.n_cores)
    for r in pool.map(worker, iterator):
        t.update(1)

    pool.close()


# /def


# ------------------------------------------------------------------------

if __name__ == "__main__":

    # call script
    main(args=None, opts=None)  # all arguments except script name


# /if


##############################################################################
# END