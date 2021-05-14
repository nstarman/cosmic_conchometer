# -*- coding: utf-8 -*-

"""Script to compute the spectral stuff.

Parameters
----------
BDmin : float (default = {BDmin})
BDmax : float (default = {BDmax})
BDstep : float (default = {BDstep})

Mmax : int (default = {Mmax})
mmax : int (default = {mmax})
lmax : int (default = {lmax})

kind : ("1F2", "2F2", "Cnogam", "Cgamma", "both") (default="both")
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

__all__ = [
    "make_parser",
    "main",
]


##############################################################################
# IMPORTS

# BUILT-IN
import argparse
import typing as T
import warnings

# PROJECT-SPECIFIC
from .spectral_Cs import main as Cs_main
from .spectral_hypergeometric import (
    _BDMAX,
    _BDMIN,
    _BDSTEP,
    _BIGMMAX,
    _LITTLEMMAX,
    _LMAX,
    _VERBOSE,
)
from .spectral_hypergeometric import DATA_DIR as _DATA_DIR
from .spectral_hypergeometric import main as hypergeometric_main
from .spectral_hypergeometric import make_parser as hypergeometric_parser

##############################################################################
# PARAMETERS

DATA_DIR = str(_DATA_DIR)

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
# Command Line
##############################################################################


def make_parser(
    *,
    BDmin: float = _BDMIN,
    BDmax: float = _BDMAX,
    BDstep: float = _BDSTEP,
    Mmax: int = _BIGMMAX,
    mmax: int = _LITTLEMMAX,
    lmax: int = _LMAX,
    # hypergeometric and Cs kind
    kind: str = "both",
    # general
    data_dir: str = DATA_DIR,
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
    # map Cgamma kinds to hypergeometric kinds.
    if kind == "both":
        pass
    elif kind == "Cnogam":
        kind = "1F2"
    elif kind == "Cgamma":
        kind = "2F2"

    parent_parser = hypergeometric_parser(
        BDmin=BDmin,
        BDmax=BDmax,
        BDstep=BDstep,
        Mmax=Mmax,
        mmax=mmax,
        lmax=lmax,
        # hypergeometric kind
        kind=kind,
        # general
        data_dir=data_dir,
        verbose=verbose,
        inheritable=True,
    )

    # make this parser, inheriting from hypergeometric_parser
    parser = argparse.ArgumentParser(
        description="Compute all the stuff.",
        add_help=not inheritable,
        conflict_handler="resolve" if not inheritable else "error",
        parents=[parent_parser],
    )

    return parser


# /def

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
    if opts is not None and args is None:
        pass
    else:
        if opts is not None:
            warnings.warn("Not using `opts` because `args` are given")
        if isinstance(args, str):
            args = args.split()

        parser = make_parser()
        opts = parser.parse_args(args)

    # /if

    # make hypergeometric cubes
    hypergeometric_main(opts=opts)

    # make C cubes
    # re-interpret "kind" argument
    if opts.kind == "both":
        pass
    elif opts.kind == "1F2":
        opts.kind = "Cnogam"
    elif opts.kind == "2F2":
        opts.kind = "Cgamma"

    Cs_main(opts=opts)


# /def

# ------------------------------------------------------------------------------

if __name__ == "__main__":

    # call script
    main(args=None, opts=None)  # all arguments except script name


# /if


##############################################################################
# END
