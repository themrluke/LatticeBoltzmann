# fluid_dynamics.pyx

# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import time
import numpy as np
cimport numpy as np

from parameters cimport Parameters
from plotting import plot_solution, setup_plot_directories


def timestep_loop(Parameters sim,
                  double[:, ::1] initial_rho,
                  double[:, :, ::1] initial_u):
    """
    Evolves the simulation over time

    Arguments:
        sim: Parameters object
        initial_rho (np.ndarray): 2D array of the fluid density at each lattice point
        initial_u (np.ndarray): 3D array of the fluid x & y velocity at each lattice point

    Returns:
        force_array (np.ndarray): Transverse force on obstacle for each timestep
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
    cdef double time_start, time_end, execution_time

    cdef str dvv_dir, streamlines_dir, test_streamlines_dir, test_mask_dir

    # Convert to memoryviews
    cdef double[::1] w = sim.w
    cdef int[:, ::1] c = sim.c
    cdef int[:, ::1] mask = sim.mask
    cdef int[:, ::1] mask2 = sim.mask2
    cdef int[::1] reflection = sim.reflection
    cdef double[:, ::1] rho = np.empty((num_x, num_y), dtype=np.float64)
    cdef double[:, :, ::1] u = np.empty((num_x, num_y, 2), dtype=np.float64)
    cdef double[:, :, ::1] feq = np.empty((num_x, num_y, num_v), dtype=np.float64)
    cdef double[:, :, ::1] f = np.empty((num_x, num_y, num_v), dtype=np.float64)
    cdef double[:, :, ::1] f_new = np.empty((num_x, num_y, num_v), dtype=np.float64)
    cdef double[:, :, ::1] temp
    cdef double[::1] force_array = np.empty((t_steps), dtype=np.float64)
    cdef double[:, :, ::1] momentum_point = np.zeros((num_x, num_y, num_v), dtype=np.float64)
 

    dvv_dir, streamlines_dir, test_streamlines_dir, test_mask_dir = setup_plot_directories()

    # Create the initial distribution by finding the equilibrium for the flow calculated above
    f = equilibrium(num_x, num_y, num_v, c, w, cs2, cs4, initial_rho, initial_u, f)
    rho = fluid_density(num_x, num_y, num_v, mask, f, rho)
    u = fluid_velocity(num_x, num_y, num_v, c, mask, f, rho, u)
    feq = equilibrium(num_x, num_y, num_v, c, w, cs2, cs4, rho, u, feq)

    # vor = fluid_vorticity(u, num_x, num_y) # Visualise the setup
    # plot_solution(sim, t=0, rho=np.asarray(rho), u=np.asarray(u), vor=vor,
    #               dvv_dir=dvv_dir,
    #               streamlines_dir=streamlines_dir, 
    #               test_streamlines_dir=test_streamlines_dir,
    #               test_mask_dir=test_mask_dir,
    #               )

    # Finally evolve the distribution in time
    time_start = time.time()
    for t in range(1, t_steps + 1):

        # Perform collision step, using the calculated density and velocity data
        f = collision(num_x, num_y, num_v, tau_inv, f, feq)

        # Streaming and reflection
        f_new, momentum_total = stream_and_reflect(
            num_x, num_y, num_v, c, mask, mask2, reflection, f, f_new, u, momentum_point
        )
        temp = f
        f = f_new
        f_new = temp

        force_array[t-1] = momentum_total # Calculate the force at current timestep

        # Calculate density and velocity data, for next time around
        rho = fluid_density(num_x, num_y, num_v, mask, f, rho)
        u = fluid_velocity(num_x, num_y, num_v, c, mask, f, rho, u)

        # Recalculate equilibrium
        feq = equilibrium(num_x, num_y, num_v, c, w, cs2, cs4, rho, u, feq)

        if (t % sim.t_plot == 0): # Visualise the simulation
            vor = fluid_vorticity(u, num_x, num_y)
            plot_solution(sim, t=t, rho=np.asarray(rho), u=np.asarray(u), vor=vor,
                          dvv_dir=dvv_dir,
                          streamlines_dir=streamlines_dir, 
                          test_streamlines_dir=test_streamlines_dir,
                          test_mask_dir=test_mask_dir)
            print(f'PLOT {t} complete')

    time_end = time.time()
    execution_time = time_end - time_start
    print(f'TIME FOR TIMESTEP_LOOP FUNCTION: {execution_time}')

    # Append the result to a text file
    with open("loop_timings.txt", "a") as file:
        file.write(f"{execution_time}\n")

    return force_array


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
        num_x (int): Lattice size in x-direction
        num_y (int): Lattice size in y-direction
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
            u_dot_u = u[i, j, 0] * u[i, j, 0] + u[i, j, 1] * u[i, j, 1]  # Magnitude squared of velocity
            for k in range(num_v):
                u_dot_c = u[i, j, 0] * c[k, 0] + u[i, j, 1] * c[k, 1] # Velocity component in direction
                feq[i, j, k] = w[k] * (1 + u_dot_c / cs2 + (u_dot_c*u_dot_c) / (2 * cs4) - u_dot_u / (2 * cs2)) * rho[i, j]

    return feq


def fluid_density(int num_x,
                  int num_y,
                  int num_v,
                  int[:, ::1] mask,
                  double[:, :, ::1] f,
                  double[:, ::1] rho):
    """
    Calculate the fluid density from the distribution function.

    Arguments:
        num_x (int): Lattice size in x-direction
        num_y (int): Lattice size in y-direction
        num_v (int): Number of velocity directions
        mask (memoryview): Binary obstacle mask
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
            if mask[i, j] == 1:  # Set fluid density inside the obstacle
                rho[i, j] = 0.0001 # To avoid divisions by 0
            else:
                total = 0.0  # Thread-safe local variable
                for k in range(num_v):
                    total = total + f[i, j, k] # Sum over all velocity directions

                rho[i, j] = total

    return rho


def fluid_velocity(int num_x,
                   int num_y,
                   int num_v,
                   int[:, ::1] c,
                   int[:, ::1] mask,
                   double[:, :, ::1] f, 
                   double[:, ::1] rho,
                   double[:, :, ::1] u):
    """
    Calculate the fluid velocity from the distribution function and fluid density.

    Arguments:
        num_x (int): Lattice size in x-direction
        num_y (int): Lattice size in y-direction
        num_v (int): Number of velocity directions
        c (memoryview): Discrete velocity directions of shape (num_v, 2)
        mask (memoryview): Binary obstacle mask
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
            if mask[i, j] == 1:
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
        num_x (int): Lattice size in x-direction
        num_y (int): Lattice size in y-direction

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
        num_x (int): Lattice size in x-direction
        num_y (int): Lattice size in y-direction
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


def stream_and_reflect(int num_x,
                       int num_y,
                       int num_v,
                       int[:, ::1] c,
                       int[:, ::1] mask,
                       int[:, ::1] mask2,
                       int[::1] reflection,
                       double[:, :, ::1] f,
                       double[:, :, ::1] f_new,
                       double[:, :, ::1] u,
                       double[:, :, ::1] momentum_point):
    """
    Perform the streaming and boundary reflection steps.

    Arguments:
        num_x (int): Lattice size in x-direction
        num_x (int): Lattice size in x-direction
        num_y (int): Lattice size in y-direction
        num_v (int): Number of velocity directions
        c (memoryview): Discrete velocity directions of shape (num_v, 2)
        mask (memoryview): Binary obstacle mask
        mask2 (memoryview): Mask region used for force calculation
        reflection (memoryview): Reflection mapping array
        f (memoryview): Distribution function array
        f_new (memoryview): Streamed distribution function array initialised as 0
        u (memoryview): 3D array of the fluid x & y velocity at each lattice point
        momentum_point (memoryview): Momentum array initialised as 0

    Returns:
        f_new (memoryview): Updated streamed distribution function array
        momentum_total (float): Total transverse force on mask2
    """

    cdef int i, j, k, rolled_x, rolled_y
    cdef double momentum_total = 0.0
    f_new[:, :, :] = 0.0

    for i in range(num_x):
        for j in range(num_y):
            for k in range(num_v):

                # Calculate the source indices for streaming
                rolled_x = (i - c[k, 0] + num_x) % num_x
                rolled_y = (j - c[k, 1] + num_y) % num_y

                # Calculate the momentum at the surface of the mask
                if mask2[i, j] == 1:
                    momentum_point[i, j, k] = 0.0

                elif mask2[rolled_x, rolled_y] == 1:
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