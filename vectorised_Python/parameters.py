# parameters.py
import numpy as np

class Parameters():
    """
    u0: 
        WHAT IS U0
            0.01 for unseparated flow
            0.06 for Foppl vortices
            0.18  for Vortex shedding
    scalemax:
        Maximum Vorticity scale value
            0.06  For Airfoil
            0.06  For Parachute
            0.015 For Cybertruck
            0.08  For one circle
            0.04  For muliple circles
    """
    def __init__(self, num_x, num_y, tau, u0, scalemax, t_steps, t_plot):
        self.num_x = num_x
        self.num_y = num_y
        self.tau = tau  # Decay timescale
        self.u0 = u0  # Initial speed
        self.nu = (2.0 * tau - 1) / 6.0  # Kinematic viscosity
        self.Re = num_x * u0 / self.nu  # Reynolds number
        self.inv_tau = 1 / tau
        self.cs = 1 / np.sqrt(3)  # Speed of sound
        self.rho0 = 1.0  # Fluid density
        self.num_vel = 9
        self.c = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1],
                           [-1, -1], [1, -1], [0, 0]])
        self.w = np.array([1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36, 4/9])
        self.reflection = np.array([2, 3, 0, 1, 6, 7, 4, 5, 8])
        self.mask = np.full((num_x, num_y), False)
        self.mask2 = np.full((num_x, num_y), False) #the part of the total mask which we are measuring the force on        
        self.scalemax = scalemax
        if scalemax == 0.015:
            self.scalemin = -0.03 #-scalemax for all shapes other than cybertruck set to: -0.03
        else:
            self.scalemin = -scalemax
        self.t_steps = t_steps # Total number of timesteps to run the simulation for
        self.t_plot = t_plot # Ploting interval (1 plot every N timesteps)