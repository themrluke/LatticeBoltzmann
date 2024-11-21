# parameters.pxd

cimport numpy as np


cdef class Parameters:
    """
    Declaration of the Cython-optimized Parameters class.
    """

    # Public attributes
    cdef public int num_x, num_y, num_v, t_steps, t_plot
    cdef public double tau, u0, nu, Re, inv_tau, cs, rho0, scalemax, scalemin
    cdef public object c
    cdef public object w
    cdef public object reflection
    cdef public object mask
    cdef public object mask2