# fluid_dynamics.py

from numba import cuda, float32
import numpy as np


@cuda.jit(fastmath=True, cache=True)
def equilibrium_kernel(rho, u, feq, c, w, cs):
    """Calculate the equilibrium distribution on the GPU."""
    i, j, k = cuda.grid(3)
    num_x, num_y, num_v = feq.shape

    if i < num_x and j < num_y and k < num_v:
        u_dot_u = u[i, j, 0] * u[i, j, 0] + u[i, j, 1] * u[i, j, 1]
        u_dot_c = u[i, j, 0] * c[k, 0] + u[i, j, 1] * c[k, 1]
        feq[i, j, k] = w[k] * (
            1 + u_dot_c / (cs*cs) +
            (u_dot_c*u_dot_c) / (2 * cs*cs*cs*cs) -
            u_dot_u / (2 * cs*cs)
        ) * rho[i, j]


@cuda.jit(fastmath=True, cache=True)
def fluid_density_kernel(f, rho, mask):
    """Calculate fluid density on the GPU."""
    i, j = cuda.grid(2)
    num_x, num_y, num_v = f.shape

    if i < num_x and j < num_y:
        if mask[i, j] == 1:
            rho[i, j] = 0.0001
        else:
            total = 0.0
            for k in range(num_v):
                total = total + f[i, j, k]
            rho[i, j] = total

@cuda.jit(fastmath=True, cache=True)
def fluid_velocity_kernel(f, rho, u, c, mask):
    """Calculate fluid velocity on the GPU."""
    i, j = cuda.grid(2)
    num_x, num_y, num_v = f.shape

    if i < num_x and j < num_y:
        if mask[i, j] == 1:
            u[i, j, 0] = 0
            u[i, j, 1] = 0
        else:
            for k in range(num_v):
                u[i, j, 0] = u[i, j, 0] + (f[i, j, k] * c[k, 0] / rho[i, j])
                u[i, j, 1] = u[i, j, 1] + (f[i, j, k] * c[k, 1] / rho[i, j])

@cuda.jit(fastmath=True, cache=True)
def fluid_vorticity_kernel(u, vor):
    """Calculate fluid vorticity on the GPU."""
    i, j = cuda.grid(2)
    num_x, num_y = vor.shape

    if i < num_x and j < num_y:
        roll_up = (i - 1 + num_x) % num_x
        roll_down = (i + 1) % num_x
        roll_left = (j - 1 + num_y) % num_y
        roll_right = (j + 1) % num_y

        vor[i, j] = (u[roll_down, j, 1] - u[roll_up, j, 1] -
                     u[i, roll_right, 0] + u[i, roll_left, 0])


@cuda.jit(fastmath=True, cache=True)
def collision_kernel(f, feq, tau):
    """Perform the collision step on the GPU."""
    i, j, k = cuda.grid(3)
    num_x, num_y, num_v = f.shape

    if i < num_x and j < num_y and k < num_v:
        f[i, j, k] = (f[i, j, k] * (1 - 1 / tau)) + (feq[i, j, k] / tau)


@cuda.jit(fastmath=True, cache=True)
def stream_and_reflect_kernel(f, f_new, momentum_point, u, mask, mask2, reflection, c, momentum_partial):
    """Perform streaming and reflection with momentum calculation."""
    i, j, k = cuda.grid(3)
    num_x, num_y, num_v = f.shape

    # Determine the total number of threads in the block
    tid = cuda.threadIdx.x + cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.z * cuda.blockDim.x * cuda.blockDim.y
    block_id = cuda.blockIdx.x + cuda.blockIdx.y * cuda.gridDim.x + cuda.blockIdx.z * cuda.gridDim.x * cuda.gridDim.y

    momentum_total = 0.0

    if i < num_x and j < num_y and k < num_v:
        rolled_x = (i - c[k, 0] + num_x) % num_x
        rolled_y = (j - c[k, 1] + num_y) % num_y

        if mask2[i, j] == 1:
            momentum_point[i, j, k] = 0.0

        elif mask2[rolled_x, rolled_y] == 1:
            momentum_point[i, j, k] = u[i, j, 0] * (f[i, j, k] + f[i, j, reflection[k]])

        else:
            momentum_point[i, j, k] = 0.0

        momentum_total = momentum_total + momentum_point[i, j, k]

        if mask[i, j] == 1:
            f_new[i, j, k] = 0.0

        elif mask[rolled_x, rolled_y] == 1:
            f_new[i, j, k] = f[i, j, reflection[k]]

        else:
            f_new[i, j, k] = f[rolled_x, rolled_y, k]

    # # Use global memory for block-wise reduction
    # momentum_partial[block_id * cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z + tid] = momentum_total
    # cuda.syncthreads()
    
    # # Perform reduction across global memory
    # if tid == 0:
    #     block_total = 0.0
    #     for t in range(cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z):
    #         block_total += momentum_partial[block_id * cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z + t]
    #     momentum_partial[block_id] = block_total