# fluid_dynamics.py

from numba import njit, prange
import numpy as np


@njit(parallel=True, fastmath=True, nogil=True, boundscheck=False, cache=True)
def equilibrium(num_x, num_y, num_v, rho, u, c, w, cs, feq):
    """
    Evaluates the equilibrium distribution across the lattice for a
    given fluid density and velocity field.

    Arguments:
        num_x (int): Lattice size in x-direction
        num_y (int): Lattice size in y-direction
        num_v (int): Number of velocity directions
        rho (np.ndarray): 2D array of the fluid density at each lattice point
        u (np.ndarray): 3D array of the fluid x & y velocity at each lattice point
        c (np.ndarray): Discrete velocity directions of shape (num_v, 2)
        w (np.ndarray): Weight coefficients for velocity directions
        cs (float): Lattice speed of sound
        feq (np.ndarray): Equilibrium distribution array initialised as 0

    Returns:
        feq (np.ndarray): Updated equilibrium distribution array
    """

    # Pre-compute the speed of sound squared & to power of 4
    cs2 = cs * cs 
    cs4 = cs2 * cs2

    for i in prange(num_x): # Parellelize over x
        for j in range(num_y):
            u_dot_u = u[i, j, 0] * u[i, j, 0] + u[i, j, 1] * u[i, j, 1] # Magnitude squared of velocity
            for k in range(num_v):
                u_dot_c = u[i, j, 0] * c[k, 0] + u[i, j, 1] * c[k, 1] # Velocity component in direction
                feq[i, j, k] = w[k] * (
                    1 + u_dot_c / cs2 +
                    (u_dot_c * u_dot_c) / (2 * cs4) -
                    u_dot_u / (2 * cs2)
                ) * rho[i, j]

    return feq


@njit(parallel=True, fastmath=True, nogil=True, boundscheck=False, cache=True)
def fluid_density(num_x, num_y, num_v, f, mask, rho):
    """
    Calculate the fluid density from the distribution function.

    Arguments:
        num_x (int): Lattice size in x-direction
        num_y (int): Lattice size in y-direction
        num_v (int): Number of velocity directions
        f (np.ndarray): Distribution function array
        mask (np.ndarray): Binary obstacle mask
        rho (np.ndarray): Density array initialised as 0

    Returns:
        rho (np.ndarray): Updated 2D array of the fluid density at each lattice point
    """

    for i in prange(num_x): # Parallelize over x
        for j in range(num_y):
            if mask[i, j] == 1: # Set fluid density inside the obstacle
                rho[i, j] = 0.0001 # To avoid divisions by 0

            else:
                total = 0.0
                for k in range(num_v):
                    total = total + f[i, j, k] # Sum over all velocity directions

                rho[i, j] = total

    return rho


@njit(parallel=True, fastmath=True, nogil=True, boundscheck=False, cache=True)
def fluid_velocity(num_x, num_y, num_v, f, rho, c, mask, u):
    """
    Calculate the fluid velocity from the distribution function and fluid density.

    Arguments:
        num_x (int): Lattice size in x-direction
        num_y (int): Lattice size in y-direction
        num_v (int): Number of velocity directions
        f (np.ndarray): Distribution function array
        rho (np.ndarray): 2D array of the fluid density at each lattice point
        c (np.ndarray): Discrete velocity directions of shape (num_v, 2)
        mask (np.ndarray): Binary obstacle mask
        u (np.ndarray): Velocity array initialised as 0

    Returns:
        u (np.ndarray): Updated 3D array of the fluid x & y velocity at each lattice point
    """

    u[:, :, :] = 0.0

    for i in prange(num_x): # Parallelize over x
        for j in range(num_y):
            if mask[i, j] == 1:
                u[i, j, :] = 0.0 # Set velocity to 0 in the obstacle

            else:
                for k in range(num_v): # Sum contributions from all velocity directions
                    u[i, j, 0] = u[i, j, 0] + (f[i, j, k] * c[k, 0] / rho[i, j])
                    u[i, j, 1] = u[i, j, 1] + (f[i, j, k] * c[k, 1] / rho[i, j])

    return u


def fluid_vorticity(u): # Uncomment for creating plots
    """
    Compute the vorticity of the velocity field.

    Arguments:
         u (np.ndarray): 3D array of the fluid x & y velocity at each lattice point

    Returns:
        vor (np.ndarray): 2D array of vorticity
    """

    vor = (np.roll(u[:, :, 1], -1, 0) - np.roll(u[:, :, 1], 1, 0) -
           np.roll(u[:, :, 0], -1, 1) + np.roll(u[:, :, 0], 1, 1))

    return vor


@njit(parallel=True, fastmath=True, nogil=True, boundscheck=False, cache=True)
def collision(num_x, num_y, num_v, f, feq, tau):
    """
    Perform the collision step, updating the distribution `f` using `feq`.

    Arguments:
        num_x (int): Lattice size in x-direction
        num_y (int): Lattice size in y-direction
        num_v (int): Number of velocity directions
        f (np.ndarray): Distribution function array
        feq (np.ndarray): Equilibrium distribution array
        tau (float): Decay timescale

    Returns:
        f (np.ndarray): Updated distribution function array
    """

    for i in prange(num_x): # Parallelize over x
        for j in range(num_y):
            for k in range(num_v):
                f[i, j, k] = (f[i, j, k] * (1 - (1/tau))) + (feq[i, j, k]/ tau)

    return f


@njit(parallel=True, fastmath=True, nogil=True, boundscheck=False, cache=True)
def stream_and_reflect(num_x, num_y, num_v, f, u, c, mask, mask2, reflection, momentum_point, f_new):
    """
    Perform the streaming and boundary reflection steps.

    Arguments:
        num_x (int): Lattice size in x-direction
        num_y (int): Lattice size in y-direction
        num_v (int): Number of velocity directions
        f (np.ndarray): Distribution function array
        u (np.ndarray): 3D array of the fluid x & y velocity at each lattice point
        c (np.ndarray): Discrete velocity directions of shape (num_v, 2) 
        mask (np.ndarray): Binary obstacle mask
        mask2 (np.ndarray): Mask region used for force calculation
        reflection (np.ndarray): Reflection mapping array
        momentum_point (np.ndarray): Momentum array initialised as 0
        f_new (np.ndarray): Streamed distribution function array initialised as 0

    Returns:
        f_new (np.ndarray): Updated streamed distribution function array
        momentum_total (float): Total transverse force on mask2

    """

    momentum_total = 0.0
    f_new = np.zeros_like(f)

    for i in prange(num_x): # Parallelize over x
        for j in range(num_y):
            for k in range(num_v):

                # Calculate the source indices for streaming
                rolled_x = (i - c[k, 0]) % num_x
                rolled_y = (j - c[k, 1]) % num_y

                # Calculate the momentum at the surface of the mask
                if mask2[i, j] == 1:
                    momentum_point[i, j, k] = 0.0

                elif mask2[rolled_x, rolled_y] == 1: # 
                    momentum_point[i, j, k] = u[i, j, 0] * (f[i, j, k] + f[i, j, reflection[k]])

                else:
                    momentum_point[i, j, k] = 0.0

                # Sum the total momentum from all points
                momentum_total += momentum_point[i, j, k]

                # Perform streaming and reflection
                if mask[i, j] == 1:
                    f_new[i, j, k] = 0.0 # No fluid inside obstacle

                elif mask[rolled_x, rolled_y] == 1:
                    f_new[i, j, k] = f[i, j, reflection[k]] # Reflection

                else:
                    f_new[i, j, k] = f[rolled_x, rolled_y, k] # Streaming

    return f_new, momentum_total