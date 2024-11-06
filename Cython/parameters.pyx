cimport numpy as np
import numpy as np

cdef class Parameters:
    """
    Cython-optimized version of the Parameters class.
    """

    def __init__(self, int num_x, int num_y, double tau, double u0, 
                 double scalemax, int t_steps, int t_plot):
        self.num_x = num_x
        self.num_y = num_y
        self.tau = tau
        self.u0 = u0
        self.nu = (2.0 * tau - 1) / 6.0
        self.Re = num_x * u0 / self.nu
        self.inv_tau = 1 / tau
        self.cs = 1 / np.sqrt(3)
        self.rho0 = 1.0
        self.num_vel = 9
        self.c = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1],
                           [-1, -1], [1, -1], [0, 0]], dtype=np.float64)
        self.w = np.array([1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36, 4/9], dtype=np.float64)
        self.reflection = np.array([2, 3, 0, 1, 6, 7, 4, 5, 8], dtype=np.int32)
        self.mask = np.full((num_x, num_y), False, dtype=np.bool_)
        self.mask2 = np.full((num_x, num_y), False, dtype=np.bool_)
        self.scalemax = scalemax
        self.scalemin = -0.03 if scalemax == 0.015 else -scalemax
        self.t_steps = t_steps
        self.t_plot = t_plot
