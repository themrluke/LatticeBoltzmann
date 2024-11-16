# fluid_dynamics.py

from numba import njit, prange
import numpy as np


# Python code with explicit loops

@njit(parallel=True, fastmath=True, nogil=True, boundscheck=False, cache=True)
def equilibrium(rho, u, num_x, num_y, num_v, c, w, cs):
    # Function for evaluating the equilibrium distribution, across the entire
    # lattice, for a given input fluid density and velocity field.

    feq = np.zeros((num_x, num_y, num_v), dtype=np.float64)

    for i in prange(num_x, ):
        for j in range(num_y):
            u_dot_u = u[i, j, 0]**2 + u[i, j, 1]**2
            for v in range(num_v):
                u_dot_c = u[i, j, 0] * c[v, 0] + u[i, j, 1] * c[v, 1]
                feq[i, j, v] = w[v] * (
                    1 + u_dot_c / cs**2 +
                    (u_dot_c**2) / (2 * cs**4) -
                    u_dot_u / (2 * cs**2)
                ) * rho[i, j]
    return feq

@njit(parallel=True, fastmath=True, nogil=True, boundscheck=False, cache=True)
def fluid_density(f, num_x, num_y, num_v, mask):
    # Calculate the fluid density from the distribution f
    rho = np.zeros((num_x, num_y), dtype=np.float64)

    for i in prange(num_x):
        for j in range(num_y):
            if mask[i, j] == 1:
                rho[i, j] = 0.0001
            else:
                total = 0.0
                for k in range(num_v):
                    total = total + f[i, j, k]
                rho[i, j] = total

    return rho # Returns a numpy array of fluid density, of shape (sim.num_x, sim.num_y)

@njit(parallel=True, fastmath=True, nogil=True, boundscheck=False, cache=True)
def fluid_velocity(f, rho, num_x, num_y, num_v, c, mask):
    # Calculate the fluid velocity from the distribution f and the fluid density rho

    u = np.zeros((num_x, num_y, 2), dtype=np.float64)

    for i in prange(num_x):
        for j in range(num_y):
            if mask[i, j] == 1:
                u[i, j, :] = 0

            else:
                for k in range(num_v):
                    u[i, j, 0] = u[i, j, 0] + (f[i, j, k] * c[k, 0] / rho[i, j])
                    u[i, j, 1] = u[i, j, 1] + (f[i, j, k] * c[k, 1] / rho[i, j])
    return u


# def fluid_vorticity(u):
#     vor = (np.roll(u[:,:,1], -1, 0) - np.roll(u[:,:,1], 1, 0) -
#            np.roll(u[:,:,0], -1, 1) + np.roll(u[:,:,0], 1, 1))
#     return vor

@njit(parallel=True, fastmath=True, nogil=True, boundscheck=False, cache=True)
def collision(f, feq, num_x, num_y, num_v, tau):
    # Perform the collision step, updating the distribution f, 
    # using the equilibrium distribution provided in feq
    for i in prange(num_x):
        for j in range(num_y):
            for k in range(num_v):
                f[i, j, k] = (f[i, j, k] * (1 - (1/tau))) + (feq[i, j, k]/ tau)
    return f


@njit(parallel=True, fastmath=True, nogil=True, boundscheck=False, cache=True)
def stream_and_reflect(f, u, num_x, num_y, num_v, c, mask, mask2, reflection):
    # Perform the streaming and boundary reflection step.
    momentum_point = np.zeros((num_x, num_y, num_v), dtype=np.float64)
    momentum_total = 0.0
    f_new = np.zeros_like(f)

    for i in prange(num_x):
        for j in range(num_y):
            for k in range(num_v):

                rolled_x = (i - c[k, 0]) % num_x
                rolled_y = (j - c[k, 1]) % num_y

                if mask2[i, j] == 1:
                    momentum_point[i, j, k] = 0.0
                
                
                elif mask2[rolled_x, rolled_y] == 1:
                    momentum_point[i, j, k] = u[i, j, 0] * (f[i, j, k] + f[i, j, reflection[k]])
        
                else:
                    momentum_point[i, j, k] = 0.0
                
                momentum_total += momentum_point[i, j, k]
                
                if mask[i, j] == 1:
                    f_new[i, j, k] = 0.0

                elif mask[rolled_x, rolled_y] == 1:
                    f_new[i, j, k] = f[i, j, reflection[k]]

                else:
                    f_new[i, j, k] = f[rolled_x, rolled_y, k]

    return f_new, momentum_total
