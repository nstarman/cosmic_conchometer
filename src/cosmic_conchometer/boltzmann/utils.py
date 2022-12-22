
"""Boltzmann Code Utilities."""


##############################################################################
# IMPORTS

# STDLIB
import functools
import typing as T


import numba
from numba import carray, cfunc, types
from scipy import LowLevelCallable

__all__ = ["llc_integrand"]

##############################################################################


def llc_integrand(
    func: T.Callable[[float, T.Tuple[float, ...]], float],
) -> LowLevelCallable:
    """Decorator to make a `scipy.LowLevelCallable` integrand from a function.

    Parameters
    ----------
    func : Callable[[float, ndarray], float]

    Returns
    -------
    `scipy.LowLevelCallable`
    """
    # inspired from http://ilovesymposia.com/2017/03/15/prettier-lowlevelcallables-with-numba-jit-and-decorators/

    # make high performance verson of the function
    jitdec = numba.jit(["double(double, double[:])"], nopython=True)
    jitfunc = jitdec(func)

    # create C signature of scipy.LowLevelCallable integrand
    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html
    c_sig = types.double(types.intc, types.CPointer(types.double))

    # make cfunc from hpc
    @cfunc(c_sig)
    def integrand(len_xargs, xargs_ptr):  # type: ignore
        """C version of integrand.

        Parameters
        ----------
        len_xargs : int
            Length of the xx array which contains xargs[0] == x and the rest
            of the items are numbers contained in the args argument of quad.
        xargs_ptr : cpointer(double)
            pointer to xargs array.

        Returns
        -------
        double
            The result.
        double
            The error.
        """
        xargs = carray(xargs_ptr, (len_xargs,), dtype=types.double)
        return jitfunc(xargs[0], xargs[1:])

    # turn into LLC
    return LowLevelCallable(integrand.ctypes)
