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

    u_dot_c = np.einsum('ijk,lk->ijl', u, sim.c)   #3D
    u_dot_u = np.einsum('ijk,ijk->ij', u, u)       #2D

    u_dot_u = u_dot_u[...,np.newaxis]
    rho = rho[..., np.newaxis] # Makes it 3D

    # Equilibrium distribution of the liquid = feq
    feq = sim.w * (1 + (u_dot_c / (sim.cs**2)) + (u_dot_c**2 / (2 * sim.cs**4)) - (u_dot_u / (2 * sim.cs**2))) * rho

    # Returns a numpy array of shape (sim.num_x, sim.num_y, sim.num_v)
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

    rho = np.where(sim.mask, 0.0001, np.einsum('ijk->ij', f))

    # Returns a numpy array of fluid density, of shape (sim.num_x, sim.num_y).
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

    inv_rho = 1 / rho
    f_over_rho = np.einsum('ijk,ij->ijk', f, inv_rho)
    u = np.where(sim.mask[..., np.newaxis], 0, np.einsum('ijk,kl->ijl', f_over_rho, sim.c))

    # Returns a numpy array of shape (sim.num_x, sim.num_y, 2), of fluid velocities
    return u


def fluid_vorticity(u):
    """
    Compute the vorticity of the velocity field.

    Arguments:
         u (np.ndarray): 3D array of the fluid x & y velocity at each lattice point

    Returns:
        vor (np.ndarray): 2D array of vorticity
    """

    vor = (np.roll(u[:, :,1], -1, 0) - np.roll(u[:, :, 1], 1, 0) -
           np.roll(u[:, :, 0], -1, 1) + np.roll(u[:, : ,0], 1, 1))

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

    delta_t = 1
    f = (f * (1 - (delta_t / sim.tau))) + (feq * (delta_t / sim.tau))

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

    delta_t = 1
    momentum_point=np.zeros((sim.num_x, sim.num_y,9))

    for i in range(len(sim.c)):
        momentum_point[:, :, i] = np.where(sim.mask2, 
                                        0, 
                                        np.where(np.roll(sim.mask2, sim.c[i, :], axis=(0, 1)), 
                                                u[:, :, 0]*(f[:, :, i] + f[:, :, sim.reflection[i]]),
                                                0))
        f[:,:,i] = np.where(sim.mask, 
                            0, 
                            np.where(np.roll(sim.mask, sim.c[i, :], axis=(0, 1)), 
                                    f[:, :, sim.reflection[i]], 
                                    np.roll(f[:, :, i], sim.c[i, :] * delta_t, axis=(0, 1))))

    momentum_total = np.einsum('ijk->',momentum_point)

    return f, momentum_total