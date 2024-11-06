# fluid_dynamics.py
import numpy as np
    
# Python code with explicit loops


def equilibrium(sim, rho, u):
    # Function for evaluating the equilibrium distribution, across the entire
    # lattice, for a given input fluid density and velocity field.
    num_x, num_y, num_v = sim.num_x, sim.num_y, len(sim.c)# num_v: Number of velocity directions

    feq = np.zeros((num_x, num_y, num_v), dtype=np.float64)

    for i in range(num_x):
        for j in range(num_y):
            u_dot_u = u[i, j, 0]**2 + u[i, j, 1]**2
            for v in range(num_v):
                u_dot_c = u[i, j, 0] * sim.c[v, 0] + u[i, j, 1] * sim.c[v, 1]
                feq[i, j, v] = sim.w[v] * (
                    1 + u_dot_c / sim.cs**2 +
                    (u_dot_c**2) / (2 * sim.cs**4) -
                    u_dot_u / (2 * sim.cs**2)
                ) * rho[i, j]
    return feq


def fluid_density(sim, f):
    # Calculate the fluid density from the distribution f
    num_x, num_y = sim.num_x, sim.num_y
    rho = np.zeros((num_x, num_y), dtype=np.float64)

    for i in range(num_x):
        for j in range(num_y):
            if sim.mask[i, j]:
                rho[i, j] = 0.0001 # was 0.0001
            else:
                rho[i, j] = np.sum(f[i, j, :])
    return rho # Returns a numpy array of fluid density, of shape (sim.num_x, sim.num_y).


def fluid_velocity(sim, f, rho):
    # Calculate the fluid velocity from the distribution f and the fluid density rho
    num_x, num_y, num_v = sim.num_x, sim.num_y, len(sim.c)
    u = np.zeros((num_x, num_y, 2), dtype=np.float64)

    for i in range(num_x):
        for j in range(num_y):
            if sim.mask[i, j]:
                u[i, j, 0] = 0
                u[i, j, 1] = 0
            else:
                for v in range(num_v):
                    u[i, j, 0] += f[i, j, v] * sim.c[v, 0]
                    u[i, j, 1] += f[i, j, v] * sim.c[v, 1]

                u[i, j, 0] /= rho[i, j]
                u[i, j, 1] /= rho[i, j]
    return u


def collision(sim, f, feq):
    # Perform the collision step, updating the distribution f, 
    # using the equilibrium distribution provided in feq
    num_x, num_y, num_v = sim.num_x, sim.num_y, len(sim.c)
    tau_inv = 1 / sim.tau
    f_new = np.zeros((num_x, num_y, num_v), dtype=np.float64)

    for i in range(num_x):
        for j in range(num_y):
            for v in range(num_v):
                f_new[i, j, v] = f[i, j, v] * (1 - tau_inv) + feq[i, j, v] * tau_inv
    return f_new


def stream_and_reflect(sim, f, u):
    # Perform the streaming and boundary reflection step.
    delta_t = 1
    num_x, num_y, num_v = sim.num_x, sim.num_y, len(sim.c)
    momentum_point = np.zeros((num_x, num_y, num_v), dtype=np.float64)

    for i in range(num_v):
        for x in range(num_x):
            for y in range(num_y):
                if sim.mask2[x, y]:
                    continue
                rolled_x = (x + sim.c[i, 0]) % num_x
                rolled_y = (y + sim.c[i, 1]) % num_y

                if sim.mask2[rolled_x, rolled_y]:
                    momentum_point[x, y, i] = u[x, y, 0] * (f[x, y, i] + f[x, y, sim.reflection[i]])
                f[x, y, i] = f[rolled_x, rolled_y, sim.reflection[i]]
    return f, np.sum(momentum_point)
