# fluid_dynamics.pyx

# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from parameters cimport Parameters
import time

def timestep_loop(Parameters sim, 
                  np.ndarray[np.float64_t, ndim=3] f, 
                  np.ndarray[np.float64_t, ndim=3] u,
                  np.ndarray[np.float64_t, ndim=3] feq):

    """
    Loop over each timestep to perform calculations
    """

    cdef int t_steps = sim.t_steps
    cdef int t
    cdef double momentum_total
    cdef np.ndarray[np.float64_t, ndim=2] rho
    cdef np.ndarray[np.float64_t, ndim=1] force_array = np.zeros((t_steps), dtype=np.float64)# Array which will store the force


    for t in range(1, t_steps + 1):
        print(f"Step {t} - f max: {np.max(f)}, f min: {np.min(f)}")
        print(f"Step {t} - u max: {np.max(u)}, u min: {np.min(u)}")

        # Perform collision step, using the calculated density and velocity data.
        time1_start = time.time()
        f = collision(sim, f, feq)
        time1_end = time.time()
        print('collision() time: ', time1_end - time1_start)

        # Streaming and reflection
        time2_start = time.time()
        f, momentum_total = stream_and_reflect(sim, f, u)
        time2_end = time.time()
        print('stream_and_reflect() time: ', time2_end - time2_start)

        force_array[t-1] = momentum_total

        # Calculate density and velocity data, for next time around
        time3_start = time.time()
        rho = fluid_density(sim, f)
        time3_end = time.time()
        print('fluid_density() time: ', time3_end - time3_start)

        time4_start = time.time()
        u = fluid_velocity(sim, f, rho)
        time4_end = time.time()
        print('fluid_velocity() time: ', time4_end - time4_start)

        time5_start = time.time()
        feq = equilibrium(sim, rho, u)
        time5_end = time.time()
        print('equilibrium() time: ', time5_end - time5_start)

    return force_array



def equilibrium(Parameters sim, 
                np.ndarray[np.float64_t, ndim=2] rho, 
                np.ndarray[np.float64_t, ndim=3] u):
    """
    Evaluate the equilibrium distribution across the lattice.
    """
    cdef int num_x = sim.num_x
    cdef int num_y = sim.num_y
    cdef int num_v = sim.num_vel
    cdef np.ndarray[np.float64_t, ndim=1] w = sim.w
    cdef np.ndarray[np.int32_t, ndim=2] c = sim.c
    cdef double cs2 = sim.cs**2
    cdef double cs4 = cs2**2
    
    cdef np.ndarray[np.float64_t, ndim=3] feq = np.zeros((num_x, num_y, num_v), dtype=np.float64)
    cdef int i, j, v
    cdef double u_dot_u, u_dot_c

    for i in range(num_x):
        for j in range(num_y):
            u_dot_u = u[i, j, 0] * u[i, j, 0] + u[i, j, 1] * u[i, j, 1]
            for v in range(num_v):
                u_dot_c = u[i, j, 0] * c[v, 0] + u[i, j, 1] * c[v, 1]
                feq[i, j, v] = w[v] * (1 + u_dot_c / cs2 + (u_dot_c*u_dot_c) / (2 * cs4) - u_dot_u / (2 * cs2)) * rho[i, j]

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
    cdef np.ndarray[np.int32_t, ndim=2] mask = sim.mask
    cdef int i, j, v
    cdef double total

    for i in range(num_x):
        for j in range(num_y):
            if mask[i, j] == 1:
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
    cdef np.ndarray[np.int32_t, ndim=2] c = sim.c
    cdef np.ndarray[np.float64_t, ndim=3] u = np.zeros((num_x, num_y, 2), dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=2] mask = sim.mask


    cdef int x, y, v
    cdef double total_x, total_y

    for x in range(num_x):
        for y in range(num_y):
            if mask[x, y] == 1:
                u[x, y, 0] = 0.0
                u[x, y, 1] = 0.0
            else:
                total_x = 0
                total_y = 0
                for v in range(num_v):
                    total_x += f[x, y, v] * c[v, 0]
                    total_y += f[x, y, v] * c[v, 1]
                u[x, y, 0] = total_x / rho[x, y]
                u[x, y, 1] = total_y / rho[x, y]
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
    cdef np.ndarray[np.int32_t, ndim=2] c = sim.c
    cdef np.ndarray[np.int32_t, ndim=1] reflection = sim.reflection
    cdef np.ndarray[np.int32_t, ndim=2] mask = sim.mask
    cdef np.ndarray[np.int32_t, ndim=2] mask2 = sim.mask2
    cdef int i, x, y, rolled_x, rolled_y
    cdef double momentum_total = 0.0

    for i in range(num_v):
        for x in range(num_x):
            for y in range(num_y):

                rolled_x = (x + c[i, 0]) % num_x
                rolled_y = (y + c[i, 1]) % num_y
                
                if mask[x, y] == 1:
                    f[x, y, i] = 0.0
                    continue


                if mask[rolled_x, rolled_y] == 1:
                    f[x, y, i] = f[x, y, sim.reflection[i]]
                else:
                    f[x, y, i] = f[rolled_x, rolled_y, i]
                
                if mask2[x, y] == 0:
                    if mask2[rolled_x, rolled_y] == 1:
                        momentum_point[x, y, i] = u[x, y, 0] * (f[x, y, i] + f[x, y, reflection[i]])
                
                momentum_total += momentum_point[x, y, i]

    return f, momentum_total