# fluid_dynamics.pyx

# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

# cython: boundscheck=True, wraparound=False, cdivision=True, initializedcheck=True

import time
import numpy as np
cimport numpy as np
from mpi4py import MPI

from parameters cimport Parameters
from plotting import plot_solution, setup_plot_directories


def timestep_loop(Parameters sim,
                  double[:, ::1] initial_rho,
                  double[:, :, ::1] initial_u,
                  int local_num_x,
                  int start_x,
                  int rank,
                  int size):
    """
    Evolves the simulation over time

    Arguments:
        sim: Parameters object
        initial_rho (np.ndarray): 2D array of the fluid density at each lattice point
        initial_u (np.ndarray): 3D array of the fluid x & y velocity at each lattice point
        local_num_x (int): Size of subdomain visible to thread
        start_x (int): Local subdomain's absolute position on the lattice
        rank (int): Current MPI rank
        size (int): Number of MPI ranks

    Returns:
        local_force_array (np.ndarray): Transverse force on obstacle for each timestep within subdomain
    """

    # Initialize parameters
    cdef int t
    cdef int t_steps = sim.t_steps
    cdef int num_x = sim.num_x
    cdef int num_y = sim.num_y
    cdef int num_v = sim.num_v
    cdef double cs = sim.cs
    cdef double cs2 = cs*cs
    cdef double cs4 = cs2*cs2
    cdef double tau_inv = sim.inv_tau
    cdef double momentum_total

    cdef str dvv_dir, streamlines_dir, test_streamlines_dir, test_mask_dir

    # Convert to memoryviews
    cdef double[::1] w = sim.w
    cdef int[:, ::1] c = sim.c
    cdef int[:, ::1] mask = sim.mask
    cdef int[:, ::1] mask2 = sim.mask2
    cdef int[::1] reflection = sim.reflection
    cdef double[:, ::1] rho = np.empty((local_num_x, num_y), dtype=np.float64)
    cdef double[:, :, ::1] u = np.empty((local_num_x, num_y, 2), dtype=np.float64)
    cdef double[:, :, ::1] feq = np.empty((local_num_x, num_y, num_v), dtype=np.float64)
    cdef double[:, :, ::1] f = np.empty((local_num_x, num_y, num_v), dtype=np.float64)
    cdef double[:, :, ::1] f_new = np.empty((local_num_x, num_y, num_v), dtype=np.float64)
    cdef double[:, :, ::1] temp
    cdef double[::1] local_force_array = np.empty((t_steps), dtype=np.float64)
    cdef double[:, :, ::1] momentum_point = np.zeros((local_num_x, num_y, num_v), dtype=np.float64)

    # Create boundarys to transfer
    cdef double[:, ::1] left_ghost = np.empty((num_y, num_v), dtype=np.float64)
    cdef double[:, ::1] right_ghost = np.empty((num_y, num_v), dtype=np.float64)


    dvv_dir, streamlines_dir, test_streamlines_dir, test_mask_dir = setup_plot_directories()

    # Create the initial distribution by finding the equilibrium for the flow calculated above
    f = equilibrium(local_num_x, num_y, num_v, c, w, cs2, cs4, initial_rho, initial_u, f)
    rho = fluid_density(local_num_x, num_y, num_v, mask, start_x, f, rho)
    u = fluid_velocity(local_num_x, num_y, num_v, c, mask, start_x, f, rho, u)
    feq = equilibrium(local_num_x, num_y, num_v, c, w, cs2, cs4, rho, u, feq)

    if rank == 0: # Visualise setup
        vor = fluid_vorticity(u, local_num_x, num_y)
        plot_solution(sim, t=0, rho=np.asarray(rho), u=np.asarray(u), vor=vor,
                    dvv_dir=dvv_dir,
                    streamlines_dir=streamlines_dir, 
                    test_streamlines_dir=test_streamlines_dir,
                    test_mask_dir=test_mask_dir,
                    )

    # Work out the rank to the left and right
    left_neighbor = rank - 1 if rank > 0 else size - 1
    right_neighbor = rank + 1 if rank < size - 1 else 0

    # Finally evolve the distribution in time
    time_start = time.time()
    for t in range(1, sim.t_steps + 1):

        # Perform collision step, using the calculated density and velocity data
        f = collision(local_num_x, num_y, num_v, tau_inv, f, feq)

        # Communicate boundaries with neighbors including periodic boundaries
        MPI.COMM_WORLD.Sendrecv(
            sendbuf=f[f.shape[0]-1, :, :], dest=right_neighbor, recvbuf=left_ghost, source=left_neighbor
        )

        MPI.COMM_WORLD.Sendrecv(
            sendbuf=f[0, :, :], dest=left_neighbor, recvbuf=right_ghost, source=right_neighbor
        )

        # Streaming and reflection
        f_new, momentum_total = stream_and_reflect(
            num_x, local_num_x, num_y, num_v, c, mask, mask2, start_x, reflection, f, f_new, u, momentum_point, left_ghost, right_ghost
        )
        temp = f
        f = f_new
        f_new = temp

        local_force_array[t - 1] = momentum_total # Calculate the force at current timestep

        # Calculate density and velocity data, for next time around
        rho = fluid_density(local_num_x, num_y, num_v, mask, start_x, f, rho)
        u = fluid_velocity(local_num_x, num_y, num_v, c, mask, start_x, f, rho, u)

        # Recalculate equilibrium
        feq = equilibrium(local_num_x, num_y, num_v, c, w, cs2, cs4, rho, u, feq)

        if rank == 0: # Visualise the simulation
            if (t % sim.t_plot == 0):
                vor = fluid_vorticity(u, local_num_x, num_y)
                plot_solution(sim, t=t, rho=np.asarray(rho), u=np.asarray(u), vor=vor,
                            dvv_dir=dvv_dir,
                            streamlines_dir=streamlines_dir, 
                            test_streamlines_dir=test_streamlines_dir,
                            test_mask_dir=test_mask_dir)
                print(f'PLOT {t} complete')

    time_end = time.time()
    print('TIME FOR TIMESTEP_LOOP FUNCTION: ', time_end - time_start)

    return local_force_array


def equilibrium(int num_x,
                int num_y,
                int num_v,
                int[:, ::1] c,
                double[::1] w,
                double cs2, double cs4,
                double[:, ::1] rho, 
                double[:, :, ::1] u,
                double[:, :, ::1] feq):
    """
    Evaluates the equilibrium distribution across the lattice for a
    given fluid density and velocity field.

    Arguments:
        num_x (int): Local lattice size in x-direction
        num_y (int): Local lattice size in y-direction
        num_v (int): Number of velocity directions
        c (memoryview): Discrete velocity directions of shape (num_v, 2)
        w (memoryview): Weight coefficients for velocity directions
        cs2, cs4 (float): Lattice speed of sound (squared & to power of 4)
        rho (memoryview): 2D array of the fluid density at each lattice point
        u (memoryview): 3D array of the fluid x & y velocity at each lattice point
        feq (memoryview): Equilibrium distribution array initialised as 0

    Returns:
        feq (memoryview): Updated equilibrium distribution array
    """

    cdef int i, j, k
    cdef double u_dot_u, u_dot_c

    feq[:, :, :] = 0.0

    for i in range(num_x):
        for j in range(num_y):
            u_dot_u = u[i, j, 0] * u[i, j, 0] + u[i, j, 1] * u[i, j, 1] # Magnitude squared of velocity
            for k in range(num_v):
                u_dot_c = u[i, j, 0] * c[k, 0] + u[i, j, 1] * c[k, 1] # Velocity component in direction
                feq[i, j, k] = w[k] * (1 + u_dot_c / cs2 + (u_dot_c * u_dot_c) / (2 * cs4) - u_dot_u / (2 * cs2)) * rho[i, j]

    return feq


def fluid_density(int num_x,
                  int num_y,
                  int num_v,
                  int[:, ::1] mask,
                  int start_x,
                  double[:, :, ::1] f,
                  double[:, ::1] rho):
    """
    Calculate the fluid density from the distribution function.

    Arguments:
        num_x (int): Local lattice size in x-direction
        num_y (int): Local lattice size in y-direction
        num_v (int): Number of velocity directions
        mask (memoryview): Binary obstacle mask
        start_x (int): Local subdomain's absolute position on the lattice
        f (memoryview): Distribution function array
        rho (memoryview): Density array initialised as 0

    Returns:
        rho (memoryview): Updated 2D array of the fluid density at each lattice point
    """

    cdef int i, j, k
    cdef double total

    rho[:, :] = 0.0

    for i in range(num_x):
        for j in range(num_y):
            if mask[i + start_x, j] == 1: # Set fluid density inside the obstacle
                rho[i, j] = 0.0001 # To avoid divisions by 0
            else:
                total = 0.0
                for k in range(num_v):
                    total = total + f[i, j, k] # Sum over all velocity directions

                rho[i, j] = total

    return rho


def fluid_velocity(int num_x,
                   int num_y,
                   int num_v,
                   int[:, ::1] c,
                   int[:, ::1] mask,
                   int start_x,
                   double[:, :, ::1] f, 
                   double[:, ::1] rho,
                   double[:, :, ::1] u):
    """
    Calculate the fluid velocity from the distribution function and fluid density.

    Arguments:
        num_x (int): Local lattice size in x-direction
        num_y (int): Local lattice size in y-direction
        num_v (int): Number of velocity directions
        c (memoryview): Discrete velocity directions of shape (num_v, 2)
        mask (memoryview): Binary obstacle mask
        start_x (int): Local subdomain's absolute position on the lattice
        f (memoryview): Distribution function array
        rho (memoryview): 2D array of the fluid density at each lattice point
        u (memoryview): Velocity array initialised as 0

    Returns:
        u (memoryview): Updated 3D array of the fluid x & y velocity at each lattice point
    """

    cdef int i, j, k
    cdef double total_x, total_y

    u[:, :, :] = 0.0

    for i in range(num_x):
        for j in range(num_y):
            if mask[i + start_x, j] == 1:
                u[i, j, :] = 0.0 # Set velocity to 0 in the obstacle
            else:
                for k in range(num_v): # Sum contributions from all velocity directions
                    u[i, j, 0] = u[i, j, 0] + (f[i, j, k] * c[k, 0] / rho[i, j])
                    u[i, j, 1] = u[i, j, 1] + (f[i, j, k] * c[k, 1] / rho[i, j])
    return u


def fluid_vorticity(double[:, :, ::1] u, int num_x, int num_y):
    """
    Compute the vorticity of the velocity field.

    Arguments:
        u (memoryview): 3D array of the fluid x & y velocity at each lattice point
        num_x (int): Local lattice size in x-direction
        num_y (int): Local lattice size in y-direction

    Returns:
        vor (memoryview): 2D array of vorticity
    """

    cdef np.ndarray[np.float64_t, ndim=2] vor = np.empty((num_x, num_y), dtype=np.float64)

    u = np.asarray(u)
    vor = (np.roll(u[:,:,1], -1, 0) - np.roll(u[:,:,1], 1, 0) -
           np.roll(u[:,:,0], -1, 1) + np.roll(u[:,:,0], 1, 1))

    return vor


def collision(int num_x,
              int num_y,
              int num_v,
              double tau_inv,
              double[:, :, ::1] f, 
              double[:, :, ::1] feq):
    """
    Perform the collision step, updating the distribution `f` using `feq`.

    Arguments:
        num_x (int): Local lattice size in x-direction
        num_y (int): Local lattice size in y-direction
        num_v (int): Number of velocity directions
        tau_inv (float): 1 / decay timescale
        f (memoryview): Distribution function array
        feq (memoryview): Equilibrium distribution array

    Returns:
        f (memoryview): Updated distribution function array
    """

    cdef int i, j, k

    for i in range(num_x):
        for j in range(num_y):
            for k in range(num_v):
                f[i, j, k] = (f[i, j, k] * (1 - tau_inv)) + (feq[i, j, k] * tau_inv)

    return f


def stream_and_reflect(int global_num_x,
                    int num_x,
                    int num_y,
                    int num_v,
                    int[:, ::1] c,
                    int[:, ::1] mask,
                    int[:, ::1] mask2,
                    int start_x,
                    int[::1] reflection,
                    double[:, :, ::1] f,
                    double[:, :, ::1] f_new,
                    double[:, :, ::1] u,
                    double[:, :, ::1] momentum_point,
                    double[:, ::1] left_ghost,
                    double[:, ::1] right_ghost):
    """
    Perform the streaming and boundary reflection steps.

    Arguments:
        golobal_num_x (int): Absolute lattice size in x-direction
        num_x (int): Local lattice size in x-direction
        num_y (int): Local lattice size in y-direction
        num_v (int): Number of velocity directions
        c (memoryview): Discrete velocity directions of shape (num_v, 2)
        mask (memoryview): Binary obstacle mask
        mask2 (memoryview): Mask region used for force calculation
        start_x (int): Local subdomain's absolute position on the lattice
        reflection (memoryview): Reflection mapping array
        f (memoryview): Distribution function array
        f_new (memoryview): Streamed distribution function array initialised as 0
        u (memoryview): 3D array of the fluid x & y velocity at each lattice point
        momentum_point (memoryview): Momentum array initialised as 0
        left_ghost (memoryview): Holds information across left boundary
        right_ghost (memoryview): Holds information across right boundary

    Returns:
        f_new (memoryview): Updated streamed distribution function array
        momentum_total (float): Total transverse force on mask2

    """

    cdef int i, j, k, rolled_x, rolled_y, wrapped_x
    cdef double momentum_total = 0.0
    f_new[:, :, :] = 0.0

    for i in range(num_x):
        for j in range(num_y):
            for k in range(num_v):

                # Calculate the source indices for streaming
                rolled_x = (i + start_x - c[k, 0] + global_num_x) % global_num_x
                rolled_y = (j - c[k, 1] + num_y) % num_y
                end_location = (i + start_x - c[k, 0])

                # Calculate the momentum at the surface of the mask
                if mask2[i + start_x, j] == 1:
                    momentum_point[i, j, k] = 0.0

                elif mask2[rolled_x, rolled_y] == 1:
                    momentum_point[i, j, k] = u[i, j, 0] * (f[i, j, k] + f[i, j, reflection[k]])

                else:
                    momentum_point[i, j, k] = 0.0

                # Sum the total momentum from all points
                momentum_total += momentum_point[i, j, k]

                # Perform streaming and reflection
                if mask[i + start_x, j] == 1:
                    f_new[i, j, k] = 0.0 # No fluid inside obstacle

                elif mask[rolled_x, rolled_y] == 1:
                    f_new[i, j, k] = f[i, j, reflection[k]] # Reflection

                elif rolled_x < start_x or (end_location == -1):
                    f_new[i, j, k] = left_ghost[rolled_y, k] # Streaming across left boundary

                elif rolled_x >= start_x + num_x or (end_location == global_num_x):
                    f_new[i, j, k] = right_ghost[rolled_y, k] # Streaming across right boundary

                else:
                    f_new[i, j, k] = f[end_location - start_x, rolled_y, k] # Streaming within boundaries

    return f_new, momentum_total