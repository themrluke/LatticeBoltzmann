# fluid_dynamics.py
import numpy as np
    
def equilibrium(sim, rho, u):
    # Function for evaluating the equilibrium distribution, across the entire
    # lattice, for a given input fluid density and velocity field.
    
    u_dot_c = np.einsum('ijk,lk->ijl', u, sim.c)   #3D
    u_dot_u = np.einsum('ijk,ijk->ij', u, u)       #2D
    
    u_dot_u = u_dot_u[...,np.newaxis]
    rho = rho[..., np.newaxis]       #makes it 3D
    
    #equilibrium distribution of the liquid = feq
    feq = sim.w*(1+(u_dot_c/(sim.cs**2))+(u_dot_c**2/(2*sim.cs**4))-(u_dot_u/(2*sim.cs**2)))*rho
    # Return a numpy array of shape (sim.num_x, sim.num_y, sim.num_v)
    return feq


def fluid_density(sim, f):
    # Calculate the fluid density from the distribution f.
    rho = np.where(sim.mask, 0.0001, np.einsum('ijk->ij', f))
    
    # Returns a numpy array of fluid density, of shape (sim.num_x, sim.num_y).
    return rho



def fluid_velocity(sim, f, rho):
    # Calculate the fluid velocity from the distribution f and the fluid
    # density rho
    
    inv_rho = 1/rho
    f_over_rho = np.einsum('ijk,ij->ijk', f, inv_rho)
    u = np.where(sim.mask[...,np.newaxis], 0, np.einsum('ijk,kl->ijl', f_over_rho, sim.c))
    # Returns a numpy array of shape (sim.num_x, sim.num_y, 2), of fluid
    # velocities.
    return u


def fluid_vorticity(u):
    vor = (np.roll(u[:,:,1], -1, 0) - np.roll(u[:,:,1], 1, 0) -
           np.roll(u[:,:,0], -1, 1) + np.roll(u[:,:,0], 1, 1))
    return vor


def collision(sim, f, feq):
    # Perform the collision step, updating the distribution f, using the
    # equilibrium distribution provided in feq.
    delta_t = 1
    f = (f*(1-(delta_t/sim.tau))) + (feq*(delta_t/sim.tau))
    
    return f


def stream_and_reflect(sim, f, u):
    # Perform the streaming and boundary reflection step.
    
    delta_t = 1
    momentum_point=np.zeros((sim.num_x,sim.num_y,9))
    
    for i in range(len(sim.c)):
        momentum_point[:,:,i] = np.where(sim.mask2, 
                                         0, 
                                         np.where(np.roll(sim.mask2, sim.c[i,:], axis=(0,1)), 
                                                  u[:,:,0]*(f[:,:,i]+f[:,:,sim.reflection[i]]),
                                                  0))               #should i roll by -sim.c as opposite direction as line below
        f[:,:,i] = np.where(sim.mask, 
                            0, 
                            np.where(np.roll(sim.mask, sim.c[i,:], axis=(0,1)), 
                                     f[:,:,sim.reflection[i]], 
                                     np.roll(f[:,:,i], sim.c[i,:]*delta_t, axis=(0,1))))
    momentum_total = np.einsum('ijk->',momentum_point)
    
    return f, momentum_total