# fluid_dynamics.py
import numpy as np
    
# Python code with explicit loops


def equilibrium(sim, rho, u):
    # Function for evaluating the equilibrium distribution, across the entire
    # lattice, for a given input fluid density and velocity field.

    feq = np.zeros((sim.num_x, sim.num_y, sim.num_v), dtype=np.float64)

    for i in range(sim.num_x):
        for j in range(sim.num_y):
            u_dot_u = u[i, j, 0]**2 + u[i, j, 1]**2
            for v in range(sim.num_v):
                u_dot_c = u[i, j, 0] * sim.c[v, 0] + u[i, j, 1] * sim.c[v, 1]
                feq[i, j, v] = sim.w[v] * (
                    1 + u_dot_c / sim.cs**2 +
                    (u_dot_c**2) / (2 * sim.cs**4) -
                    u_dot_u / (2 * sim.cs**2)
                ) * rho[i, j]
    return feq

def fluid_density(sim, f):
    # Calculate the fluid density from the distribution f
    rho = np.zeros((sim.num_x, sim.num_y), dtype=np.float64)

    for i in range(sim.num_x):
        for j in range(sim.num_y):
            if sim.mask[i, j] == 1:
                rho[i, j] = 0.0001
            else:
                total = 0.0
                for k in range(sim.num_v):
                    total = total + f[i, j, k]
                rho[i, j] = total

    return rho # Returns a numpy array of fluid density, of shape (sim.num_x, sim.num_y)


def fluid_velocity(sim, f, rho):
    # Calculate the fluid velocity from the distribution f and the fluid density rho

    u = np.zeros((sim.num_x, sim.num_y, 2), dtype=np.float64)

    for i in range(sim.num_x):
        for j in range(sim.num_y):
            if sim.mask[i, j] == 1:
                u[i, j, :] = 0

            else:
                for k in range(sim.num_v):
                    u[i, j, 0] = u[i, j, 0] + (f[i, j, k] * sim.c[k, 0] / rho[i, j])
                    u[i, j, 1] = u[i, j, 1] + (f[i, j, k] * sim.c[k, 1] / rho[i, j])
    return u


def fluid_vorticity(u):
    vor = (np.roll(u[:,:,1], -1, 0) - np.roll(u[:,:,1], 1, 0) -
           np.roll(u[:,:,0], -1, 1) + np.roll(u[:,:,0], 1, 1))
    return vor


def collision(sim, f, feq):
    # Perform the collision step, updating the distribution f, 
    # using the equilibrium distribution provided in feq
    for i in range(sim.num_x):
        for j in range(sim.num_y):
            for k in range(sim.num_v):
                f[i, j, k] = (f[i, j, k] * (1 - (1/sim.tau))) + (feq[i, j, k]/ sim.tau)
    return f



def stream_and_reflect(sim, f, u):
    # Perform the streaming and boundary reflection step.
    momentum_point = np.zeros((sim.num_x, sim.num_y, sim.num_v), dtype=np.float64)
    momentum_total = 0.0
    f_new = np.zeros_like(f)

    for i in range(sim.num_x):
        for j in range(sim.num_y):
            for k in range(sim.num_v):

                rolled_x = (i - sim.c[k, 0]) % sim.num_x
                rolled_y = (j - sim.c[k, 1]) % sim.num_y

                if sim.mask2[i, j] == 1:
                    momentum_point[i, j, k] = 0.0
                
                else:
                    if sim.mask2[rolled_x, rolled_y] == 1:
                        momentum_point[i, j, k] = u[i, j, 0] * (f[i, j, k] + f[i, j, sim.reflection[k]])
            
                    else:
                        momentum_point[i, j, k] = 0.0
                
                momentum_total += momentum_point[i, j, k]
                
                if sim.mask[i, j] == 1:
                    f_new[i, j, k] = 0.0
                    continue

                elif sim.mask[rolled_x, rolled_y] == 1:
                    f_new[i, j, k] = f[i, j, sim.reflection[k]]
                else:
                    f_new[i, j, k] = f[rolled_x, rolled_y, k]
                


    return f_new, momentum_total
