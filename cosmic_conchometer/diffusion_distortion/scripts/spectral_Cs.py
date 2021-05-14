# -*- coding: utf-8 -*-

"""Script to compute the 'C' cubes used in the spectral calculations.

Must have already calculated the hypergeometric cubes!

Parameters
----------
kind : ("Cnogam", "Cgamma", "both") (default="both")
    Whether to calculate the C w/(out) cubes or both.

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
import itertools
import pathlib
import typing as T
import warnings
from collections.abc import Generator, Sequence

# THIRD PARTY
import numpy as np
import schwimmbad
import zarr
from scipy.special import gamma
from scipy.special import spherical_jn as besselJ

# PROJECT-SPECIFIC
from .spectral_hypergeometric import _VERBOSE, DATA_DIR
from cosmic_conchometer.setup_package import HAS_TQDM, _NoOpPBar

# from mpmath import mpc, mpf  # gamma


if HAS_TQDM:
    # THIRD PARTY
    from tqdm import tqdm


__all__ = [
    "make_parser",
    "main",
    # functions
    "Cnogam_from_1F2",
    "Cgamma_from_2F2",
]

##############################################################################
# PARAMETERS

# format documentation
__doc__.format(data_dir=DATA_DIR, verbose=_VERBOSE)

##############################################################################
# CODE
##############################################################################


def Cnogam_from_1F2(F: np.ndarray, bd: float) -> np.ndarray:
    r"""Calculate C1F2 cube.

    Parameters
    ----------
    F : (l, m, M) ndarray
    bd : float
        :math:`\beta\Delta`

    Returns
    -------
    C : (l-1, m, M) ndarray

    """
    bd = float(bd)
    lenl, lenm, lenM = F.shape
    ls, ms, Ms = np.mgrid[0 : lenl - 1, 0:lenm, 0:lenM]
    factor = np.power(ls / (ls + 1), Ms)  # TODO! check stability

    if bd == 0:
        t1 = 0
    else:
        t1 = np.power(bd, -ms - 1) * (
            besselJ(ms + 1, bd * (ls + 1)) - factor * besselJ(ms + 1, bd * ls)
        )

    t2 = (
        np.sqrt(np.pi)
        * (Ms - (2 * ms ** 2 + 6 * ms + 5))
        / (4 * (Ms + ms + 1) * gamma(ms + 5 / 2))
        * np.power(2.0, -ms)
    )
    # TODO! simplify? what's the danger of almost cancelations?
    t3 = (
        np.power(ls + 1, ms + 1) * F[1:]
        - factor * np.power(ls, ms + 1) * F[:-1]
    )  # the l+1 1F2 - the l 1F2

    return t1 - t2 * t3


# /def


def Cgamma_from_2F2(F: np.ndarray, bd: float) -> np.ndarray:
    r"""Calculate C2F2 cube.

    .. todo::

        use mpmath functions if contents of F are mpmath

    Parameters
    ----------
    F : (l, m, M) ndarray
    bd : float
        :math:`\beta\Delta`

    Returns
    -------
    C : (l-1, m, M) ndarray

    """
    bd = float(bd)
    lenl, lenm, lenM = F.shape
    ls, ms, Ms = np.mgrid[0 : lenl - 1, 0:lenm, 0:lenM]
    factor = np.power(ls / (ls + 1), Ms)  # TODO! check stability

    if bd == 0:
        t1 = 0
    else:
        t1 = np.power(bd, -(ms + 1)) * (
            besselJ(ms + 1, bd * (ls + 1)) * np.exp(1j * bd * (ls + 1))
            - factor * besselJ(ms + 1, bd * ls) * np.exp(1j * bd * ls)
        )

    t2 = np.sqrt(np.pi) / np.power(2, ms + 2) / gamma(ms + 2.5)

    # TODO! simplify? what's the danger of almost cancelations?
    t3 = np.power(ls + 1, ms + 2) * bd * 1j / (Ms + ms + 2) * F[1:]
    t5 = factor * np.power(ls, ms + 2) * bd * 1j / (Ms + ms + 2) * F[:-1]
    t4 = (
        np.power(ls + 1, ms + 1)
        * (2 * ms ** 2 + 6 * ms - Ms + 5)
        / (Ms + ms + 1)
        * F[1:]
    )
    t6 = (
        factor
        * np.power(ls, ms + 1)
        * (2 * ms ** 2 + 6 * ms - Ms + 5)
        / (Ms + ms + 1)
        * F[:-1]
    )

    raise t1 - t2 * (t3 - t4 - t5 + t6)


# /def

# ------------------------------------------------------------------------------
# Read info from F cubes


def _from_numpy_dir(
    path: pathlib.Path,
) -> tuple[Sequence[float], Generator, tuple]:
    fs = tuple(path.glob("*.npy"))
    """Load from numpy directory."""
    bDs = [float(f.stem.split("-")[-1].replace("_", ".")) for f in fs]
    iterator = (
        ("Cnogam", np.load(f, allow_pickle=True), bD) for bD, f in zip(bDs, fs)
    )
    shape = (len(bDs), *np.load(fs[0], allow_pickle=True).shape)

    return bDs, iterator, shape


# /def


def _from_zarr_dir(
    path: pathlib.Path,
) -> tuple[Sequence[float], Generator, tuple]:
    """Load from zarr directory."""
    rFs = zarr.open(str(path), mode="r")
    bDs = tuple(rFs.attrs["betaDelta"])
    iterator = (rFs[i] for i, bD in enumerate(bDs))
    shape = rFs.shape

    return bDs, iterator, shape


# /def

##############################################################################
# Command Line
##############################################################################


def make_parser(
    *,
    data_dir: str = str(DATA_DIR),
    verbose: bool = _VERBOSE,
    inheritable: bool = False,
) -> argparse.ArgumentParser:
    r"""Expose `~argparse.ArgumentParser` for :func:`~.main`.

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
        description="Compute 'C' cubes for spectral calculations.",
        add_help=not inheritable,
        conflict_handler="resolve" if not inheritable else "error",
    )

    # Cnogam (1F2) or Cgamma (2F2)
    parser.add_argument(
        "--kind",
        choices=["Cnogam", "Cgamma", "both"],
        type=str,
        default="both",
    )

    # where stuff is saved and where to save it
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

    return parser


# /def


# ------------------------------------------------------------------------------


class Worker:
    """Worker."""

    def __init__(self, kind: str, shape: tuple, bDs: Sequence) -> None:

        if kind in ("Cnogam", "both"):
            self.Cnogam_drct = DATA_DIR / "scriptC_nogam.zarr"
            self.Cnogam = zarr.open(
                str(self.Cnogam_drct),
                mode="a",  # read/write (create if doesn’t exist)
                shape=shape,
                chunks=(3, *shape[1:]),
                dtype=np.float128,
            )
            self.Cnogam.attrs["betaDelta"] = np.array(bDs)

        if kind in ("Cgamma", "both"):
            self.Cgamma_drct = DATA_DIR / "scriptC_gamma.zarr"
            self.Cgamma = zarr.open(
                str(self.Cgamma_drct),
                mode="a",  # read/write (create if doesn’t exist)
                shape=shape,
                chunks=(3, *shape[1:]),
                dtype=np.complex128,
            )
            self.Cnogam.attrs["betaDelta"] = np.array(bDs)

    # /def

    def __call__(
        self,
        task: tuple[str, T.Callable, np.float128],
    ) -> np.complex128:
        kind, F, bD = task

        if kind == "Cnogam":
            arr = self.Cnogam
            func = Cnogam_from_1F2
        elif kind == "Cgamma":
            arr = self.Cgamma
            func = Cgamma_from_2F2
        else:
            raise ValueError("`kind` must be one of {'Cnogam', 'Cgamma'}")

        # calculate and store
        idx = arr.attrs["betaDelta"].index(bD)  # exact match!
        arr[idx] = func(F, bD)  # zarr auto-saves

    # /def


# /class

# ------------------------------------------------------------------------------


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
            warnings.warn("Not using `opts` because `args` are given")
        if isinstance(args, str):
            args = args.split()

        parser = make_parser()
        p = parser.parse_args(args)

    # /if

    iterators = []  # accumulate. then compress.
    if p.kind in ("Cnogam", "both"):
        path = pathlib.Path(p.data_dir) / "hyp1f2"
        if path.exists():  # it's a numpy
            bDs, itr, shape = _from_numpy_dir(path)
        elif path.with_suffix(".zarr").exists():
            bDs, itr, shape = _from_zarr_dir(path.with_suffix(".zarr"))
        else:
            raise IOError("trouble with 1F2")
        iterators.append(itr)

    if p.kind in ("Cgamma", "both"):
        path = pathlib.Path(p.data_dir) / "hy21f2"
        if path.exists():  # it's a numpy
            bDs, itr, shape = _from_numpy_dir(path)
        elif path.with_suffix(".zarr").exists():
            bDs, itr, shape = _from_zarr_dir(path.with_suffix(".zarr"))
        else:
            raise IOError("trouble with 2F2")
        iterators.append(itr)

    iterator: T.Iterator = itertools.chain.from_iterable(iterators)
    # compress iterators into single iterator.

    worker = Worker(p.kind, shape, bDs)
    pool = schwimmbad.choose_pool(mpi=p.mpi, processes=p.n_cores)
    t = tqdm(total=len(bDs)) if (p.verbose and HAS_TQDM) else _NoOpPBar()
    for r in pool.map(worker, iterator):
        t.update(1)

    pool.close()


# /def

# ------------------------------------------------------------------------------

if __name__ == "__main__":

    # call script
    main(args=None, opts=None)  # all arguments except script name


# /if


##############################################################################
# END
