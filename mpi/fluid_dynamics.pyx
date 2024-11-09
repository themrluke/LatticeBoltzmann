# fluid_dynamics.pyx

# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False, wraparound=False

import numpy as np
cimport numpy as np
from mpi4py import MPI
from parameters cimport Parameters
import time

def timestep_loop(Parameters sim,
                  double[:, :] initial_rho,
                  double[:, :, :] initial_u,
                  int local_num_x,
                  int start_x,
                  int end_x,
                  int rank,
                  int size):
    """
    Loop over each timestep for the local subdomain.
    """
    # Initialize parameters
    cdef int t
    cdef int num_y = sim.num_y
    cdef int num_v = sim.num_v
    cdef int[:, :] c = sim.c
    cdef int[:, :] mask = sim.mask
    cdef int[:, :] mask2 = sim.mask2
    cdef int[:] reflection = sim.reflection
    cdef double[:] w = sim.w
    cdef double momentum_total
    cdef double cs = sim.cs
    cdef double cs2 = cs * cs
    cdef double cs4 = cs2 * cs2
    cdef double tau_inv = sim.inv_tau

    # Create local arrays
    cdef double[:, :] left_ghost = np.empty((num_y, num_v), dtype=np.float64)
    #print(f"Rank {rank}: Initialized left_ghost with shape {left_ghost.shape} Rank {rank}: num_y = {type(num_y)}, num_v = {type(num_v)}")
    cdef double [:, :] right_ghost = np.empty((num_y, num_v), dtype=np.float64)
    cdef double[:] local_force_array = np.zeros(sim.t_steps, dtype=np.float64)
    cdef double[:, :] rho = np.empty((local_num_x, num_y), dtype=np.float64)
    #print(f"Rank {rank}: Initialized rho with shape {rho.shape}")
    cdef double[:, :, :] feq = np.empty((local_num_x, num_y, num_v), dtype=np.float64)
    #print(f"Rank {rank}: Initialized feq with shape {feq.shape}")
    cdef double[:, :, :] u = np.empty((local_num_x, num_y, 2), dtype=np.float64)
    cdef double[:, :, :] f = np.empty((local_num_x, num_y, num_v), dtype=np.float64)
    cdef double[:, :, :] momentum_point = np.empty((local_num_x, num_y, num_v), dtype=np.float64)
    momentum_point[:, :, :] = 0.0   


    # Initialize local subdomain
    f = equilibrium(local_num_x, num_y, num_v, c, w, cs, cs2, cs4, initial_rho, initial_u, f)

    rho = fluid_density(local_num_x, num_y, num_v, mask, f, rho)
    u = fluid_velocity(local_num_x, num_y, num_v, c, mask, f, rho, u)
    feq = equilibrium(local_num_x, num_y, num_v, c, w, cs, cs2, cs4, rho, u, feq)

    for t in range(1, sim.t_steps + 1):
        print(f"Step {t} - f max: {np.max(f)}, f min: {np.min(f)}")
        print(f"Step {t} - u max: {np.max(u)}, u min: {np.min(u)}")
        print(f"Step {t}: Mass = {np.sum(rho)}")

        
        #print(f"Rank {rank}: Starting timestep {t}")

        # Communicate boundaries with neighbors or periodic wrapping
        if rank % 2 == 0:
            if rank > 0:
                #print(f"Rank {rank}: Preparing to send to left {rank - 1}")
                MPI.COMM_WORLD.Sendrecv(sendbuf=f[0, :, :], dest=rank - 1,
                                        recvbuf=left_ghost, source=rank - 1)
                #print(f"Rank {rank}: Sent to left {rank - 1}, received from left {rank - 1}")
            else:
                #print(f"Rank {rank}: Preparing to send to left {size - 1}")
                MPI.COMM_WORLD.Sendrecv(sendbuf=f[0, :, :], dest=size - 1,
                                        recvbuf=left_ghost, source=size - 1)
                #print(f"Rank {rank}: Sent to left {size - 1}, received from left {size - 1}")

        else:
            if rank < size - 1:
                #print(f"Rank {rank}: Preparing to send to right {rank + 1}")
                MPI.COMM_WORLD.Sendrecv(sendbuf=f[-1, :, :], dest=rank + 1,
                                        recvbuf=right_ghost, source=rank + 1)
                #print(f"Rank {rank}: Sent to right {rank + 1}, received from right {rank + 1}")
            else:
                #print(f"Rank {rank}: Preparing to send to right 0")
                MPI.COMM_WORLD.Sendrecv(sendbuf=f[-1, :, :], dest=0,
                                        recvbuf=right_ghost, source=0)
                #print(f"Rank {rank}: Sent to right 0, received from right 0")

        #MPI.COMM_WORLD.Barrier()
        #print(f"Rank {rank}: Passed barrier after communication at timestep {t}")

        # Collision step
        f = collision(local_num_x, num_y, num_v, tau_inv, f, feq)

        # Streaming and reflection
        f, momentum_total = stream_and_reflect(
            local_num_x, num_y, num_v, c, mask[start_x:end_x], mask2[start_x:end_x],
            reflection, f, u, momentum_point, left_ghost, right_ghost)

        # Update force array
        local_force_array[t - 1] = momentum_total

        # Update density and velocity
        rho = fluid_density(local_num_x, num_y, num_v, mask[start_x:end_x], f, rho)
        u = fluid_velocity(local_num_x, num_y, num_v, c, mask[start_x:end_x], f, rho, u)

        # Update equilibrium
        feq = equilibrium(local_num_x, num_y, num_v, c, w, cs, cs2, cs4, rho, u, feq)

    return local_force_array




def equilibrium(int num_x,
                int num_y,
                int num_v,
                int[:, :] c,
                double[:] w,
                double cs, double cs2, double cs4,
                double[:, :] rho, 
                double[:, :, :] u,
                double[:, :, :] feq):
    """
    Evaluate the equilibrium distribution across the lattice.
    """

    cdef int i, j, k
    cdef double u_dot_u, u_dot_c

    for i in range(num_x):
        for j in range(num_y):
            u_dot_u = u[i, j, 0] * u[i, j, 0] + u[i, j, 1] * u[i, j, 1]
            for k in range(num_v):
                u_dot_c = u[i, j, 0] * c[k, 0] + u[i, j, 1] * c[k, 1]
                feq[i, j, k] = w[k] * (1 + u_dot_c / cs2 + (u_dot_c*u_dot_c) / (2 * cs4) - u_dot_u / (2 * cs2)) * rho[i, j]

    return feq


def fluid_density(int num_x,
                  int num_y,
                  int num_v,
                  int[:, :] mask,
                  double[:, :, :] f,
                  double[:, :] rho):
    """
    Calculate fluid density from the distribution f.
    """

    cdef int i, j, k
    cdef double total

    for i in range(num_x):
        for j in range(num_y):
            if mask[i, j] == 1:
                rho[i, j] = 0.0001
            else:
                total = 0.0  # Thread-safe local variable
                for k in range(num_v):
                    total = total + f[i, j, k]
                rho[i, j] = total
    return rho


def fluid_velocity(int num_x,
                   int num_y,
                   int num_v,
                   int[:, :] c,
                   int[:, :] mask,
                   double[:, :, :] f, 
                   double[:, :] rho,
                   double[:, :, :] u):
    """
    Calculate fluid velocity from the distribution f and density rho.
    """

    cdef int i, j, k
    cdef double total_x, total_y

    for i in range(num_x):  # Parallelize over x
        for j in range(num_y):
            if mask[i, j] == 1:
                u[i, j, 0] = 0.0
                u[i, j, 1] = 0.0
            else:
                total_x = 0.0  # Declare inside the inner loop
                total_y = 0.0  # Declare inside the inner loop

                for k in range(num_v):
                    total_x = total_x + (f[i, j, k] * c[k, 0])
                    total_y = total_y + (f[i, j, k] * c[k, 1])
                u[i, j, 0] = total_x / rho[i, j]
                u[i, j, 1] = total_y / rho[i, j]

                
    return u

def collision(int num_x,
              int num_y,
              int num_v,
              double tau_inv,
              double[:, :, :] f, 
              double[:, :, :] feq):
    """
    Perform the collision step, updating the distribution f using feq.
    """

    cdef int i, j, k

    for i in range(num_x):
        for j in range(num_y):
            for k in range(num_v):
                f[i, j, k] = f[i, j, k] * (1 - tau_inv) + feq[i, j, k] * tau_inv
    return f


def stream_and_reflect(int num_x,
                       int num_y,
                       int num_v,
                       int[:, :] c,
                       int[:, :] mask,
                       int[:, :] mask2,
                       int[:] reflection,
                       double[:, :, :] f, 
                       double[:, :, :] u,
                       double[:, :, :] momentum_point,
                       double[:, :] left_ghost,
                       double[:, :] right_ghost):
    """
    Perform the streaming and boundary reflection step.
    """

    cdef int i, j, k, rolled_x, rolled_y, wrapped_x
    cdef double momentum_total = 0.0

    for i in range(num_x):
        for j in range(num_y):
            for k in range(num_v):

                rolled_x = (i + c[k, 0])
                rolled_y = (j + c[k, 1]) % num_y
                
                if mask[i, j] == 1:
                    f[i, j, k] = 0.0
                    continue

                # Handle particles streaming out of the local subdomain
                if rolled_x < 0:
                    # Use the left ghost cell for particles leaving to the left
                    f[i, j, k] = left_ghost[j, k]
                    continue

                elif rolled_x >= num_x:
                    # Use the right ghost cell for particles leaving to the right
                    f[i, j, k] = right_ghost[j, k]
                    continue

                elif mask[rolled_x, rolled_y] == 1:
                    # Reflect at obstacles within the subdomain
                    f[i, j, k] = f[i, j, reflection[k]]

                else:
                    # Regular streaming within subdomain
                    f[i, j, k] = f[rolled_x, rolled_y, k]
                
                if mask2[i, j] == 0:
                    if mask2[rolled_x, rolled_y] == 1:
                        momentum_point[i, j, k] = u[i, j, 0] * (f[i, j, k] + f[i, j, reflection[k]])
                
                momentum_total += momentum_point[i, j, k]

    return f, momentum_total