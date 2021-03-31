# -*- coding: utf-8 -*-

"""**DOCSTRING**.

This script can be run from the command line with the following parameters:

Parameters
----------

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

# THIRD PARTY
import tqdm
import numpy as np
from scipy.special import gamma
from scipy.special import spherical_jn as besselJ
from numpy import power, exp, sqrt, pi
from mpmath import hyp1f2, hyp2f2

##############################################################################
# PARAMETERS

# General
_PLOT: bool = True  # Whether to plot the output
_VERBOSE: bool = True  # Degree of logfile verbosity

# Specific
_BDMIN: float = 0.0
_BDMAX: float = 200.0
_BDSTEP: float = 1.0

_BIGMMAX: int = 100
_LITTLEMMAX: int = 30
_LMAX: int = 10  # 9 + 1

##############################################################################
# CODE
##############################################################################


@np.vectorize
def _scriptC_component(
    betaDelta: float, M: int, m: int, l: int
) -> float:
    """Eqn 115 of https://www.overleaf.com/project/5efe491b4140390001b1c892

    .. math::

        \frac{1}{\beta\Delta} \Bigg[\tilde{\rho}_{ES}^{M} \Exp{i\beta\Delta\tilde{\rho}_{ES}} j_{m+1}(\beta\Delta \tilde{\rho}_{ES})\Bigg]
        - \frac{\sqrt{\pi } \tilde{\rho}_{ES}^{M+1} (\beta\Delta \tilde{\rho}_{ES})^m}{2^{m+2}  \Gamma \left(\frac{2m\!+\!5}{2}\right)}
        \Bigg[
            \frac{i\beta\Delta \tilde{\rho}_{ES}}{(M\!+\!m\!+\!2)} \, {_2F_2}(m\!+\!2,m\!+\!M\!+\!2;2m\!+\!4,m\!+\!M\!+\!3;2i \beta\Delta \tilde{\rho}_{ES})
            -\frac{2m^2\!+\!6m\!-\!M\!+\!5}{(M\!+\!m\!+\!1)} \, {_2F_2}(m\!+\!2,m\!+\!M\!+\!1;2m\!+\!4,m\!+\!M\!+\!2;2i \beta\Delta \tilde{\rho}_{ES})
        \Bigg]

    Parameters
    ----------
    betaDelta:
        .. math::

            \beta = |\vec{k}| \lambda_0 \cos{\theta_{kS}}
            \Delta = \frac{1}{N} \sqrt(\frac{(1+1/a_{eq})}{2})

    M : int
    m : int
    l : int

    Returns
    -------
    float

    """
    x = betaDelta * l

    t1 = exp(1j * x) * besselJ(m + 1, x) / power(betaDelta, m + 1)
    t2 = sqrt(pi) / power(2, m + 2) * power(l, m + 1) / gamma(m + 2.5)
    t3 = (
        (1j * x)
        / (M + m + 2)
        * hyp2f2(m + 2, m + M + 2, 2 * m + 4, m + M + 3, 2j * x)
    )
    # (2m^2+6m-M+5) / (M+m+1) * hyp2f2(m+2, m+M+1, 2m+4, m+M+2, 2i x)
    t4 = (
        (2 * m ** 2 + 6 * m - M + 5)
        / (M + m + 1)
        * hyp2f2(m + 2, m + M + 1, 2 * m + 4, m + M + 2, 2j * x)
    )

    return t1 - t2 * (t3 - t4)


# /def

##############################################################################
# Command Line
##############################################################################


def make_parser(
    *,
    bDmin: float = _BDMIN,
    bDmax: float = _BDMAX,
    bDstep: float = _BDSTEP,
    Mmax: int = _BIGMMAX,
    mmax: int = _LITTLEMMAX,
    lmax: int = _LMAX,
    # general
    plot: bool = _PLOT,
    verbose: bool = _VERBOSE,
    inheritable: bool = False,
) -> argparse.ArgumentParser:
    """Expose ArgumentParser for ``main``.

    Parameters
    ----------
    inheritable: bool, optional, keyword only
        whether the parser can be inherited from (default False).
        if True, sets ``add_help=False`` and ``conflict_hander='resolve'``

    plot : bool, optional, keyword only
        Whether to produce plots, or not.

    verbose : bool, optional, keyword only
        Script logging verbosity.

    Returns
    -------
    parser: |ArgumentParser|
        The parser with arguments:

        - plot
        - verbose

    ..
      RST SUBSTITUTIONS

    .. |ArgumentParser| replace:: `~argparse.ArgumentParser`

    """
    parser = argparse.ArgumentParser(
        description="",
        add_help=not inheritable,
        conflict_handler="resolve" if not inheritable else "error",
    )
    
    parser.add_argument("--bDmin", action="store", default=bDmin, type=float)
    parser.add_argument("--bDmax", action="store", default=bDmax, type=float)
    parser.add_argument("--bDstep", action="store", default=bDstep, type=float)

    parser.add_argument("--Mmax", action="store", default=Mmax, type=int)
    parser.add_argument("--mmax", action="store", default=mmax, type=int)
    parser.add_argument("--lmax", action="store", default=lmax, type=int)

    # plot or not
    parser.add_argument("--plot", action="store", default=plot, type=bool)
    # script verbosity
    parser.add_argument(
        "-v", "--verbose", action="store", default=verbose, type=bool
    )

    return parser


# /def


# ------------------------------------------------------------------------


def main(
    args: T.Union[list, str, None] = None,
    opts: T.Optional[argparse.Namespace] = None,
):
    """Script Function.

    Parameters
    ----------
    args : list or str or None, optional
        an optional single argument that holds the sys.argv list,
        except for the script name (e.g., argv[1:])
    opts : `~argparse.Namespace`| or None, optional
        pre-constructed results of parsed args
        if not None, used ONLY if args is None

    """
    nsp: argparse.Namespace
    if opts is not None and args is None:
        nsp = opts
    else:
        if opts is not None:
            warnings.warn("Not using `opts` because `args` are given")
        if isinstance(args, str):
            args = args.split()

        parser = make_parser()
        nsp = parser.parse_args(args)

    # /if

    betaDeltas = np.arange(nsp.bDmin, nsp.bDmax, nsp.bDstep)
    lgrid, mgrid, Mgrid = np.mgrid[0:nsp.lmax, 0:nsp.mmax, 0:nsp.Mmax]
    
    # TODO! verbosity
    for bD in tqdm.tqdm(betaDeltas):
    
        grid = _scriptC_component(bD, M=Mgrid, m=mgrid, l=lgrid)
        
        # TODO! a save-to location
        np.save(f"scriptC_comp_{bD}", grid)

# /def


# ------------------------------------------------------------------------

if __name__ == "__main__":

    # call script
    main(args=None, opts=None)  # all arguments except script name


# /if


##############################################################################
# END
