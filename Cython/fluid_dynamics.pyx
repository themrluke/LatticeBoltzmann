import numpy as np
cimport numpy as np
from parameters cimport Parameters


def equilibrium(Parameters sim, 
                np.ndarray[np.float64_t, ndim=2] rho, 
                np.ndarray[np.float64_t, ndim=3] u):
    """
    Evaluate the equilibrium distribution across the lattice.
    """
    cdef int num_x = sim.num_x
    cdef int num_y = sim.num_y
    cdef int num_v = sim.num_vel
    cdef double cs2 = sim.cs**2
    cdef double cs4 = cs2**2
    cdef np.ndarray[np.float64_t, ndim=3] feq = np.zeros((num_x, num_y, num_v), dtype=np.float64)
    cdef int i, j, v
    cdef double u_dot_u, u_dot_c

    for i in range(num_x):
        for j in range(num_y):
            u_dot_u = u[i, j, 0]**2 + u[i, j, 1]**2
            for v in range(num_v):
                u_dot_c = u[i, j, 0] * sim.c[v, 0] + u[i, j, 1] * sim.c[v, 1]
                feq[i, j, v] = sim.w[v] * (
                    1 + u_dot_c / cs2 +
                    (u_dot_c**2) / (2 * cs4) -
                    u_dot_u / (2 * cs2)
                ) * rho[i, j]
    return feq


def fluid_density(Parameters sim, 
                  np.ndarray[np.float64_t, ndim=3] f):
    """
    Calculate fluid density from the distribution f.
    """
    cdef int num_x = sim.num_x
    cdef int num_y = sim.num_y
    cdef int num_v = sim.num_vel
    cdef np.ndarray[np.float64_t, ndim=2] rho = np.zeros((num_x, num_y), dtype=np.float64)
    cdef int i, j, v
    cdef double total

    for i in range(num_x):
        for j in range(num_y):
            if sim.mask[i, j]:
                rho[i, j] = 0.0001
            else:
                total = 0
                for v in range(num_v):
                    total += f[i, j, v]
                rho[i, j] = total
    return rho


def fluid_velocity(Parameters sim, 
                   np.ndarray[np.float64_t, ndim=3] f, 
                   np.ndarray[np.float64_t, ndim=2] rho):
    """
    Calculate fluid velocity from the distribution f and density rho.
    """
    cdef int num_x = sim.num_x
    cdef int num_y = sim.num_y
    cdef int num_v = sim.num_vel
    cdef np.ndarray[np.float64_t, ndim=3] u = np.zeros((num_x, num_y, 2), dtype=np.float64)
    cdef int i, j, v
    cdef double total_x, total_y

    for i in range(num_x):
        for j in range(num_y):
            if sim.mask[i, j]:
                u[i, j, 0] = 0
                u[i, j, 1] = 0
            else:
                total_x = 0
                total_y = 0
                for v in range(num_v):
                    total_x += f[i, j, v] * sim.c[v, 0]
                    total_y += f[i, j, v] * sim.c[v, 1]
                u[i, j, 0] = total_x / rho[i, j]
                u[i, j, 1] = total_y / rho[i, j]
    return u

def collision(Parameters sim, 
              np.ndarray[np.float64_t, ndim=3] f, 
              np.ndarray[np.float64_t, ndim=3] feq):
    """
    Perform the collision step, updating the distribution f using feq.
    """
    cdef int num_x = sim.num_x
    cdef int num_y = sim.num_y
    cdef int num_v = sim.num_vel
    cdef double tau_inv = sim.inv_tau
    cdef np.ndarray[np.float64_t, ndim=3] f_new = np.zeros((num_x, num_y, num_v), dtype=np.float64)
    cdef int i, j, v

    for i in range(num_x):
        for j in range(num_y):
            for v in range(num_v):
                f_new[i, j, v] = f[i, j, v] * (1 - tau_inv) + feq[i, j, v] * tau_inv
    return f_new


def stream_and_reflect(Parameters sim, 
                       np.ndarray[np.float64_t, ndim=3] f, 
                       np.ndarray[np.float64_t, ndim=3] u):
    """
    Perform the streaming and boundary reflection step.
    """
    cdef int delta_t = 1
    cdef int num_x = sim.num_x
    cdef int num_y = sim.num_y
    cdef int num_v = sim.num_vel
    cdef np.ndarray[np.float64_t, ndim=3] momentum_point = np.zeros((num_x, num_y, num_v), dtype=np.float64)
    cdef int i, x, y, rolled_x, rolled_y
    cdef double momentum

    for i in range(num_v):
        for x in range(num_x):
            for y in range(num_y):
                if sim.mask2[x, y]:
                    continue

                # Apply periodic boundary conditions
                rolled_x = (x + int(sim.c[i, 0])) % num_x
                rolled_y = (y + int(sim.c[i, 1])) % num_y

                # Handle streaming and momentum calculation
                if sim.mask2[rolled_x, rolled_y]:
                    momentum = u[x, y, 0] * (f[x, y, i] + f[x, y, sim.reflection[i]])
                    momentum_point[x, y, i] = momentum
                else:
                    momentum_point[x, y, i] = 0.0

                # Update the distribution f
                f[x, y, i] = f[rolled_x, rolled_y, sim.reflection[i]]

    # Calculate total momentum
    cdef double momentum_total = 0.0
    for x in range(num_x):
        for y in range(num_y):
            for i in range(num_v):
                momentum_total += momentum_point[x, y, i]

    return f, momentum_total