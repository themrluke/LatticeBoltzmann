# parameters.pyx

cimport numpy as np
import numpy as np
from libc.math cimport sqrt


cdef class Parameters:
    """
    Cython-optimized class to initialise the parameters for the Lattice Boltzmann simulation.

    Attributes:
        num_x (int): Lattice size in x-direction
        num_y (int): Lattice size in y-direction
        tau (float): Decay timescale
        u0 (float): Initial fluid velocity
            Recommended values:
                - 0.01 for unseparated flow
                - 0.06 for Foppl vortices
                - 0.18  for Vortex shedding
        scalemax (float): Maximum vorticity scale for plotting
            Recommended values:
                - 0.06  For Airfoil
                - 0.06  For Parachute
                - 0.015 For Cybertruck
                - 0.08  For one circle
                - 0.04  For muliple circles
        t_steps (int): Number of timesteps to run simulation for
        t_plot (int): Number of timesteps between plots
    """

    def __init__(self, int num_x, int num_y, double tau, double u0, 
                 double scalemax, int t_steps, int t_plot):

        self.num_x = num_x # Lattice size in x
        self.num_y = num_y # Lattice size in y
        self.tau = tau  # Decay timescale
        self.u0 = u0  # Initial speed
        self.nu = (2.0 * tau - 1) / 6.0  # Kinematic viscosity
        self.Re = num_x * u0 / self.nu  # Reynolds number
        self.inv_tau = 1 / tau
        self.cs = 1 / np.sqrt(3)  # Speed of sound
        self.rho0 = 1.0  # Initial fluid density
        self.num_v = 9 # Number of velocity directions
        self.c = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1],
                           [-1, -1], [1, -1], [0, 0]], dtype=np.int32) # D2Q9 model discrete velocity directions
        self.w = np.array([1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36, 4/9]) # Weight coefficients for each velocity direction
        self.reflection = np.array([2, 3, 0, 1, 6, 7, 4, 5, 8], dtype=np.int32) # Reflection mapping
        self.mask = np.full((num_x, num_y), 0, dtype=np.int32) # Initialise the binary obstacle mask
        self.mask2 = np.full((num_x, num_y), 0, dtype=np.int32) # Region of total mask to measure the force on

        self.scalemax = scalemax # Maximum vorticity scale for plots
        if scalemax == 0.015:
            self.scalemin = -0.03 # For cybertruck
        else:
            self.scalemin = -scalemax # For all shapes other than cybertruck

        self.t_steps = t_steps # Total number of timesteps to run the simulation for
        self.t_plot = t_plot # Ploting interval