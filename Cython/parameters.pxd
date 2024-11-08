# parameters.pxd

cimport numpy as np

cdef class Parameters:
    """
    Declaration of the Cython-optimized Parameters class.
    """

    # Public attributes
    cdef public int num_x, num_y, num_vel, t_steps, t_plot
    cdef public double tau, u0, nu, Re, inv_tau, cs, rho0, scalemax, scalemin
    cdef public object c  # Velocity directions (will hold np.ndarray)
    cdef public object w  # Weights (will hold np.ndarray)
    cdef public object reflection  # Reflection indices (will hold np.ndarray)
    cdef public object mask  # Boolean mask (will hold np.ndarray)
    cdef public object mask2  # Force measurement mask (will hold np.ndarray)

