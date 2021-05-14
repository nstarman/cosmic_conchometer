# -*- coding: utf-8 -*-

"""Script to compute the hypergeometric cubes used in the spectral calculations.

This script can be run from the command line with the following parameters:

Parameters
----------
BDmin : float (default = {BDmin})
BDmax : float (default = {BDmax})
BDstep : float (default = {BDstep})

Mmax : int (default = {Mmax})
mmax : int (default = {mmax})
lmax : int (default = {lmax})

kind : ("1F2", "2F2", "both") (default="both")
    Whether to calculate the 1f2 or 2f2 hypergeometric function cubes or both.

Other Parameters
----------------
ncores : int (default=1)
    Number of processes (uses multiprocessing). Argument into :mod:`schwimmbad`.
mpi : bool (default=False)
    Run with MPI. Argument into :mod:`schwimmbad`.

data_dir : str (default={data_dir})
    The data directory in which the cubes are saved.

verbose : bool (default={verbose})
    Whether to show the progress bar.

"""

##############################################################################
# IMPORTS

from __future__ import annotations

# BUILT-IN
import argparse
import pathlib
import typing as T
import warnings

# THIRD PARTY
import numpy as np
import schwimmbad
from mpmath import hyp1f2, hyp2f2, mpc, mpf

# PROJECT-SPECIFIC
from cosmic_conchometer.data import DATA_DIR
from cosmic_conchometer.setup_package import HAS_TQDM, _NoOpPBar

# OPTIONAL DEPENDENCIES
if HAS_TQDM:
    # THIRD PARTY
    from tqdm import tqdm

__all__ = [
    "make_parser",
    "main",
    # functions
    "hypergeometric_1f2",
    "hypergeometric_2f2",
]

##############################################################################
# PARAMETERS

# General
_VERBOSE: bool = True  # Degree of logfile verbosity

# Specific
_BDMIN: float = 0.0
_BDMAX: float = 200.0
_BDSTEP: float = 1.0

_BIGMMAX: int = 100
_LITTLEMMAX: int = 30
_LMAX: int = 50  # 49 + 1

# format documentation
__doc__.format(
    BDmin=_BDMIN,
    BDmax=_BDMAX,
    BDstep=_BDSTEP,
    Mmax=_BIGMMAX,
    mmax=_LITTLEMMAX,
    lmax=_LMAX,
    data_dir=DATA_DIR,
    verbose=_VERBOSE,
)

##############################################################################
# CODE
##############################################################################


@np.vectorize
def hypergeometric_1f2(
    betaDelta: mpf,
    M: int,
    m: int,
    rhoES: int,
    astype: T.Optional[type] = None,
) -> T.Union[mpf, np.float]:
    r"""Hypergeometric 1f2 for the spectral distortion calculation.

    Parameters
    ----------
    betaDelta : `~mpmath.mpf`
        .. math::

            \beta = |\vec{k}| \lambda_0 \cos{\theta_{kS}}
            \Delta = \frac{1}{N} \sqrt(\frac{(1+1/a_{eq})}{2})

    M, m : int
        arguments.

        .. todo:: rename m vs M. too confusing.

    rhoES : int

    Returns
    -------
    `~mpmath.mpc`

    """
    astype = astype or mpf
    f = hyp1f2(
        (M + m + 1) / 2,
        m + 2.5,
        (M + m + 3) / 2,
        -((betaDelta * rhoES / 2) ** 2),
    )
    return f.astype(astype)


# /def


@np.vectorize
def hypergeometric_2f2(
    betaDelta: mpf,
    M: int,
    m: int,
    rhoES: int,
    astype: T.Optional[type] = None,
) -> T.Union[mpc, np.complex]:
    r"""Hypergeometric 2f2

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
    f : `~mpmath.mpc`
        Has a real and imaginary component because the argument to the
        :math:`_2F_2` is complex.

    """
    astype = astype or mpc
    f = hyp2f2(m + 2, M + m + 1, 2 * m + 4, m + M + 2, 2j * betaDelta * rhoES)
    return f.astype(astype)


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
    # hypergeometric kind
    kind: str = "both",
    # general
    data_dir: str = str(DATA_DIR),
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
    # make parser
    parser = argparse.ArgumentParser(
        description="Compute hypergeometric cubes for spectral calculations.",
        add_help=not inheritable,
        conflict_handler="resolve" if not inheritable else "error",
    )

    # beta-delta arguments
    parser.add_argument("--BDmin", action="store", default=BDmin, type=float)
    parser.add_argument("--BDmax", action="store", default=BDmax, type=float)
    parser.add_argument("--BDstep", action="store", default=BDstep, type=float)

    # indices argmunets
    parser.add_argument("--Mmax", action="store", default=Mmax, type=int)
    parser.add_argument("--mmax", action="store", default=mmax, type=int)
    parser.add_argument("--lmax", action="store", default=lmax, type=int)

    # 1F2 or 2F2
    parser.add_argument(
        "--kind",
        choices=["1F2", "2F2", "both"],
        type=str,
        default="both",
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

    def __init__(self, opts: argparse.Namespace) -> None:
        # this defines the cube for all terms in the sums to perform the
        # diffusion distortion integral.
        grid = np.mgrid[0 : opts.lmax, 0 : opts.mmax, 0 : opts.Mmax]
        self.L, self.m, self.M = grid
        # TODO! rename m vs M. too confusing

        self.verbose = opts.verbose

        drct = pathlib.Path(opts.data_dir)
        drct.mkdir(exist_ok=True)

        # Hypergeometric 1f2
        oneFtwo_folder = "hyp1f2"
        oneFtwo_drct = drct.joinpath(oneFtwo_folder)
        oneFtwo_drct.mkdir(exist_ok=True)
        self.oneFtwo_drct = oneFtwo_drct

        # Hypergeometric 2f2
        twoFtwo_folder = "hyp2f2"
        twoFtwo_drct = drct.joinpath(twoFtwo_folder)
        twoFtwo_drct.mkdir(exist_ok=True)
        self.twoFtwo_drct = twoFtwo_drct

    # /def

    def __call__(
        self,
        task: T.Union[tuple[mpf, str], tuple[mpf, str, T.Optional[type]]],
    ) -> T.Union[mpc, np.ndarray]:
        """Compute and save.

        Parameters
        ----------
        bD : float
        kind : {"1F2", "2F2"}
            Whether to calculate the 1F2 or 2F2 hypergeometric function.
        astype : None or type

        Returns
        -------
        F : array-like [mpf, mpc, np.float, np.complex]
            The 1F2 or 2F2 hypergeometric calculation.

        """
        astype = None if (len(task) == 2) else task[-1]
        bDstr = str(task[0]).replace(".", "_")

        # compute C
        if task[1] == "1F2":
            F = hypergeometric_2f2(
                mpf(task[0]),
                M=self.M,
                m=self.m,
                rhoES=self.L,
                astype=astype,
            )
            np.save(self.twoFtwo_drct / ("hyp2f2-" + bDstr), F)

        elif task[1] == "2F2":
            F = hypergeometric_1f2(
                mpf(task[0]),
                M=self.M,
                m=self.m,
                rhoES=self.L,
                astype=astype,
            )
            np.save(self.oneFtwo_drct / ("hyp1f2-" + bDstr), F)
        else:
            raise ValueError("`kind` must be one of {'1f2', '2f2'}")

        return F

    # /def


# /class

# ------------------------------------------------------------------------


def main(
    args: T.Union[list, str, None] = None,
    opts: T.Optional[argparse.Namespace] = None,
) -> None:
    """Script Function.

    Parameters
    ----------
    args : list or str or None (optional)
        an optional single argument that holds the sys.argv list,
        except for the script name (e.g., argv[1:])
    opts : `~argparse.Namespace` or None (optional)
        pre-constructed results of parsed args
        if not None, used ONLY if args is None

    """
    p: argparse.Namespace
    if opts is not None and args is None:
        p = opts
    else:
        if opts is not None:
            warnings.warn("Not using `opts` because `args` are given.")
        if isinstance(args, str):
            args = args.split()

        parser = make_parser()
        p = parser.parse_args(args)

    # /if

    # make the range of $\beta \Delta$ values over which to evaluate.
    # this is the independent and continuous variable and the result will
    # need to be interpolated as a function of $\beta \Delta$.
    betaDeltas = np.arange(p.BDmin, p.BDmax, p.BDstep)

    # the iterator provides the beta-delta value and the hypergeometric type
    # to calculate (1F2 or 2F2).
    if p.kind == "both":  # make alternating iterator
        iterator = ((bD, kind) for bD in betaDeltas for kind in ("1F2", "2F2"))
        leniter = 2 * len(betaDeltas)
    else:
        iterator = ((bD, p.kind) for bD in betaDeltas)
        leniter = len(betaDeltas)

    t = tqdm(total=leniter) if (p.verbose and HAS_TQDM) else _NoOpPBar()

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
