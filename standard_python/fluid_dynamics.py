# fluid_dynamics.py

import numpy as np


def equilibrium(sim, rho, u):
    """
    Evaluates the equilibrium distribution across the lattice for a
    given fluid density and velocity field.

    Arguments:
        sim: Parameters object
        rho (np.ndarray): 2D array of the fluid density at each lattice point
        u (np.ndarray): 3D array of the fluid x & y velocity at each lattice point

    Returns:
        feq (np.ndarray): Updated equilibrium distribution array
    """

    feq = np.zeros((sim.num_x, sim.num_y, sim.num_v), dtype=np.float64)

    for i in range(sim.num_x):
        for j in range(sim.num_y):
            u_dot_u = u[i, j, 0]**2 + u[i, j, 1]**2 # Magnitude squared of velocity
            for v in range(sim.num_v):
                u_dot_c = u[i, j, 0] * sim.c[v, 0] + u[i, j, 1] * sim.c[v, 1] # Velocity component in direction
                feq[i, j, v] = sim.w[v] * (
                    1 + u_dot_c / sim.cs**2 +
                    (u_dot_c**2) / (2 * sim.cs**4) -
                    u_dot_u / (2 * sim.cs**2)
                ) * rho[i, j]

    return feq


def fluid_density(sim, f):
    """
    Calculate the fluid density from the distribution function.

    Arguments:
        sim: Parameters object
        f (np.ndarray): Distribution function array

    Returns:
        rho (np.ndarray): Updated 2D array of the fluid density at each lattice point
    """

    rho = np.zeros((sim.num_x, sim.num_y), dtype=np.float64)

    for i in range(sim.num_x):
        for j in range(sim.num_y):
            if sim.mask[i, j] == 1: # Set fluid density inside the obstacle
                rho[i, j] = 0.0001 # To avoid divisions by 0
            else:
                total = 0.0
                for k in range(sim.num_v):
                    total = total + f[i, j, k]
                rho[i, j] = total

    return rho


def fluid_velocity(sim, f, rho):
    """
    Calculate the fluid velocity from the distribution function and fluid density.

    Arguments:
        sim: Parameters object
        f (np.ndarray): Distribution function array
        rho (np.ndarray): 2D array of the fluid density at each lattice point

    Returns:
        u (np.ndarray): Updated 3D array of the fluid x & y velocity at each lattice point
    """

    u = np.zeros((sim.num_x, sim.num_y, 2), dtype=np.float64)

    for i in range(sim.num_x):
        for j in range(sim.num_y):
            if sim.mask[i, j] == 1:
                u[i, j, :] = 0 # Set velocity to 0 in the obstacle

            else:
                for k in range(sim.num_v):
                    u[i, j, 0] = u[i, j, 0] + (f[i, j, k] * sim.c[k, 0] / rho[i, j])
                    u[i, j, 1] = u[i, j, 1] + (f[i, j, k] * sim.c[k, 1] / rho[i, j])

    return u


def fluid_vorticity(u):
    """
    Compute the vorticity of the velocity field.

    Arguments:
         u (np.ndarray): 3D array of the fluid x & y velocity at each lattice point

    Returns:
        vor (np.ndarray): 2D array of vorticity
    """

    vor = (np.roll(u[:,:,1], -1, 0) - np.roll(u[:,:,1], 1, 0) -
           np.roll(u[:,:,0], -1, 1) + np.roll(u[:,:,0], 1, 1))

    return vor


def collision(sim, f, feq):
    """
    Perform the collision step, updating the distribution `f` using `feq`.

    Arguments:
        sim: Parameters object
        f (np.ndarray): Distribution function array
        feq (np.ndarray): Equilibrium distribution array

    Returns:
        f (np.ndarray): Updated distribution function array
    """

    for i in range(sim.num_x):
        for j in range(sim.num_y):
            for k in range(sim.num_v):
                f[i, j, k] = (f[i, j, k] * (1 - (1/sim.tau))) + (feq[i, j, k]/ sim.tau)

    return f



def stream_and_reflect(sim, f, u):
    """
    Perform the streaming and boundary reflection steps.

    Arguments:
        sim: Parameters object
        f (np.ndarray): Distribution function array
        u (np.ndarray): 3D array of the fluid x & y velocity at each lattice point

    Returns:
        f_new (np.ndarray): Updated streamed distribution function array
        momentum_total (float): Total transverse force on mask2
    """

    momentum_point = np.zeros((sim.num_x, sim.num_y, sim.num_v), dtype=np.float64)
    momentum_total = 0.0
    f_new = np.zeros_like(f)

    for i in range(sim.num_x):
        for j in range(sim.num_y):
            for k in range(sim.num_v):

                # Calculate the source indices for streaming
                rolled_x = (i - sim.c[k, 0]) % sim.num_x
                rolled_y = (j - sim.c[k, 1]) % sim.num_y

                # Calculate the momentum at the surface of the mask
                if sim.mask2[i, j] == 1:
                    momentum_point[i, j, k] = 0.0

                elif sim.mask2[rolled_x, rolled_y] == 1:
                    momentum_point[i, j, k] = u[i, j, 0] * (f[i, j, k] + f[i, j, sim.reflection[k]])

                else:
                    momentum_point[i, j, k] = 0.0

                # Sum the total momentum from all points
                momentum_total += momentum_point[i, j, k]

                # Perform streaming and reflection
                if sim.mask[i, j] == 1:
                    f_new[i, j, k] = 0.0 # No fluid inside obstacle

                elif sim.mask[rolled_x, rolled_y] == 1:
                    f_new[i, j, k] = f[i, j, sim.reflection[k]] # Reflection

                else:
                    f_new[i, j, k] = f[rolled_x, rolled_y, k] # Streaming

    return f_new, momentum_total
