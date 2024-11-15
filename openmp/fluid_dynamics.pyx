# fluid_dynamics.pyx

# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

# cython: boundscheck=False, wraparound=False


import numpy as np
cimport numpy as np
from cython.parallel import prange
from parameters cimport Parameters
from plotting import plot_solution, setup_plot_directories
import time

def timestep_loop(Parameters sim,
                  double[:, :] initial_rho,
                  double[:, :, :] initial_u):

    """
    Loop over each timestep to perform calculations
    """

    cdef int t
    cdef int t_steps = sim.t_steps
    cdef int num_x = sim.num_x
    cdef int num_y = sim.num_y
    cdef int num_v = sim.num_v
    cdef double cs = sim.cs
    cdef double cs2 = cs*cs
    cdef double cs4 = cs2*cs2
    cdef double tau_inv = sim.inv_tau

    cdef double[:] w = sim.w
    cdef int[:, :] c = sim.c
    cdef int[:, :] mask = sim.mask
    cdef int[:, :] mask2 = sim.mask2
    cdef int[:] reflection = sim.reflection
    cdef double[:, :] rho = np.empty((num_x, num_y), dtype=np.float64)
    cdef double[:, :, :] u = np.empty((num_x, num_y, 2), dtype=np.float64)
    cdef double[:, :, :] feq = np.empty((num_x, num_y, num_v), dtype=np.float64)
    cdef double[:, :, :] f = np.empty((num_x, num_y, num_v), dtype=np.float64)
    cdef double[:, :, :] f_new = np.empty((num_x, num_y, num_v), dtype=np.float64)
    cdef double[:, :, :] temp
    cdef double[:] force_array = np.empty((t_steps), dtype=np.float64)
    cdef double[:, :, :] momentum_point = np.empty((num_x, num_y, num_v), dtype=np.float64)
    momentum_point[:, :, :] = 0.0
    cdef double momentum_total

    cdef str dvv_dir, streamlines_dir, test_streamlines_dir, test_mask_dir

    dvv_dir, streamlines_dir, test_streamlines_dir, test_mask_dir = setup_plot_directories()


    # Create the initial distribution by finding the equilibrium for the flow
    # calculated above.
    f = equilibrium(num_x, num_y, num_v, c, w, cs, cs2, cs4, initial_rho, initial_u, f)

    rho = fluid_density(num_x, num_y, num_v, mask, f, rho)
    u = fluid_velocity(num_x, num_y, num_v, c, mask, f, rho, u)
    feq = equilibrium(num_x, num_y, num_v, c, w, cs, cs2, cs4, rho, u, feq)

    vor = fluid_vorticity(u, num_x, num_y)

    plot_solution(sim, t=0, rho=np.asarray(rho), u=np.asarray(u), vor=vor,
                  dvv_dir=dvv_dir,
                  streamlines_dir=streamlines_dir, 
                  test_streamlines_dir=test_streamlines_dir,
                  test_mask_dir=test_mask_dir,
                  )



    for t in range(1, t_steps + 1):
        #print(f"Step {t} - f max: {np.max(f)}, f min: {np.min(f)}")
        #print(f"Step {t} - u max: {np.max(u)}, u min: {np.min(u)}")

        # Perform collision step, using the calculated density and velocity data.
        time1_start = time.time()
        f = collision(num_x, num_y, num_v, tau_inv, f, feq)
        time1_end = time.time()
        #print('collision() time: ', time1_end - time1_start)

        # Streaming and reflection
        time2_start = time.time()
        f_new, momentum_total = stream_and_reflect(num_x, num_y, num_v, c, mask, mask2, reflection, f, f_new, u, momentum_point)
        temp = f
        f = f_new
        f_new = temp
        time2_end = time.time()
        #print('stream_and_reflect() time: ', time2_end - time2_start)

        force_array[t-1] = momentum_total

        # Calculate density and velocity data, for next time around
        time3_start = time.time()
        rho = fluid_density(num_x, num_y, num_v, mask, f, rho)
        time3_end = time.time()
        #print('fluid_density() time: ', time3_end - time3_start)

        time4_start = time.time()
        u = fluid_velocity(num_x, num_y, num_v, c, mask, f, rho, u)
        time4_end = time.time()
        #print('fluid_velocity() time: ', time4_end - time4_start)

        time5_start = time.time()
        feq = equilibrium(num_x, num_y, num_v, c, w, cs, cs2, cs4, rho, u, feq)
        time5_end = time.time()
        #print('equilibrium() time: ', time5_end - time5_start)

        if (t % sim.t_plot == 0):
            vor = fluid_vorticity(u, num_x, num_y)
            plot_solution(sim, t=t, rho=np.asarray(rho), u=np.asarray(u), vor=vor,
                          dvv_dir=dvv_dir,
                          streamlines_dir=streamlines_dir, 
                          test_streamlines_dir=test_streamlines_dir,
                          test_mask_dir=test_mask_dir)

    return force_array



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

    feq[:, :, :] = 0.0

    for i in prange(num_x, nogil=True, schedule="static"):
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

    rho[:, :] = 0.0

    for i in prange(num_x, nogil=True, schedule="static"):
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

    u[:, :, :] = 0.0

    for i in prange(num_x, nogil=True, schedule="static"):  # Parallelize over x
        for j in range(num_y):
            if mask[i, j] == 1:     # Is this needed
                u[i, j, :] = 0.0
            else:
                for k in range(num_v):
                    u[i, j, 0] = u[i, j, 0] + (f[i, j, k] * c[k, 0] / rho[i, j])
                    u[i, j, 1] = u[i, j, 1] + (f[i, j, k] * c[k, 1] / rho[i, j])

    return u

def fluid_vorticity(double[:, :, :] u, int num_x, int num_y):

    cdef np.ndarray[np.float64_t, ndim=2] vor = np.empty((num_x, num_y), dtype=np.float64)

    u = np.asarray(u)
    vor = (np.roll(u[:,:,1], -1, 0) - np.roll(u[:,:,1], 1, 0) -
           np.roll(u[:,:,0], -1, 1) + np.roll(u[:,:,0], 1, 1))
    return vor

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

    for i in prange(num_x, nogil=True, schedule="static"):
        for j in range(num_y):
            for k in range(num_v):
                f[i, j, k] = (f[i, j, k] * (1 - tau_inv)) + (feq[i, j, k] * tau_inv)
    return f


def stream_and_reflect(int num_x,
                       int num_y,
                       int num_v,
                       int[:, :] c,
                       int[:, :] mask,
                       int[:, :] mask2,
                       int[:] reflection,
                       double[:, :, :] f,
                       double[:, :, :] f_new,
                       double[:, :, :] u,
                       double[:, :, :] momentum_point):
    """
    Perform the streaming and boundary reflection step.
    """

    cdef int i, j, k, rolled_x, rolled_y
    cdef double momentum_total = 0.0
    f_new[:, :, :] = 0.0

    for i in prange(num_x, nogil=True, schedule="static"):
        for j in range(num_y):
            for k in range(num_v):

                rolled_x = (i - c[k, 0]) % num_x # Negative because our c vector point away from lattice node
                rolled_y = (j - c[k, 1]) % num_y

                if mask2[i, j] == 1:
                    momentum_point[i, j, k] = 0.0
                
                elif mask2[rolled_x, rolled_y] == 1:
                    momentum_point[i, j, k] = u[i, j, 0] * (f[i, j, k] + f[i, j, reflection[k]])
            
                else:
                    momentum_point[i, j, k] = 0.0
                
                momentum_total += momentum_point[i, j, k]
                
                if mask[i, j] == 1:
                    f_new[i, j, k] = 0.0

                elif mask[rolled_x, rolled_y] == 1:
                    f_new[i, j, k] = f[i, j, reflection[k]]

                else:
                    f_new[i, j, k] = f[rolled_x, rolled_y, k]

    return f_new, momentum_total