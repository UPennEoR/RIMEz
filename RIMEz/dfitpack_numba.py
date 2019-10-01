import os

import numpy as np
import numba as nb
import cffi

ffi = cffi.FFI()

current_dir = os.path.dirname(os.path.realpath(__file__))
libpath = os.path.join(current_dir, "dfitpack_wrappers/dfitpack_wrappers.so")

dfitpack_wrappers_lib = ffi.dlopen(libpath)

ffi.cdef(
    """
void bispeu_wrap(double *tx, int *nx, double *ty, int *ny,
                double *c, int *kx, int *ky,
                double *x, double *y, double *z, int *m,
                double *wrk, int *lwrk, int *ier);
"""
)

bispeu_wrap = dfitpack_wrappers_lib.bispeu_wrap


@nb.njit
def bispeu_nb(tx, ty, c, kx, ky, x, y):
    """
    Evaluates a spline defined by scipy.interpolate.RectBivariateSpline.

    This function is intended to provide exactly the same output as the
    evaluation method of RectBivariateSpline, but in the form of a
    numba.njit -decorated function which can be used inside other numba
    compiled functions.

    Parameters:
    ----------
    tx, ty : float 1d-array
        The knot arrays computed by RectBivariateSpline
    c : float 1d-array
        The array of spline coefficeints computed by RectBivariateSpline
    kx, ky : int
        The order of the spline in each dimension.
    x, y : float 1d-array
        Flat arrays of coordinates (x[i], y[i]) at which to evaluate the spline.
    """
    nx = np.array([len(tx)], dtype=nb.intc)
    ny = np.array([len(ty)], dtype=nb.intc)
    kx = np.array([kx], dtype=nb.intc)
    ky = np.array([ky], dtype=nb.intc)

    m = np.array([len(x)], dtype=nb.intc)
    z = np.empty((len(x)), dtype=nb.float64)

    wrk = np.empty((kx[0] + ky[0] + 2), dtype=nb.float64)
    lwrk = np.array([len(wrk)], dtype=nb.intc)

    ier = np.empty((1,), dtype=nb.intc)

    bispeu_wrap(
        ffi.from_buffer(tx),
        ffi.from_buffer(nx),
        ffi.from_buffer(ty),
        ffi.from_buffer(ny),
        ffi.from_buffer(c),
        ffi.from_buffer(kx),
        ffi.from_buffer(ky),
        ffi.from_buffer(x),
        ffi.from_buffer(y),
        ffi.from_buffer(z),
        ffi.from_buffer(m),
        ffi.from_buffer(wrk),
        ffi.from_buffer(lwrk),
        ffi.from_buffer(ier),
    )

    return z, ier[0]
