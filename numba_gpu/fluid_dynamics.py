# fluid_dynamics.py

from numba import cuda, float64
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

    # Shared memory for block-wise reduction
    shared_momentum = cuda.shared.array(1024, dtype=float64)  # Adjust size as needed

    # Thread ID within the block
    tid = cuda.threadIdx.x + cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.z * cuda.blockDim.x * cuda.blockDim.y
    block_id = cuda.blockIdx.x + cuda.blockIdx.y * cuda.gridDim.x + cuda.blockIdx.z * cuda.gridDim.x * cuda.gridDim.y

    local_momentum = 0.0

    if i < num_x and j < num_y and k < num_v:
        rolled_x = (i - c[k, 0] + num_x) % num_x
        rolled_y = (j - c[k, 1] + num_y) % num_y

        if mask2[i, j] == 1:
            momentum_point[i, j, k] = 0.0

        elif mask2[rolled_x, rolled_y] == 1:
            momentum_point[i, j, k] = u[i, j, 0] * (f[i, j, k] + f[i, j, reflection[k]])

        else:
            momentum_point[i, j, k] = 0.0

        local_momentum = local_momentum + momentum_point[i, j, k]

        if mask[i, j] == 1:
            f_new[i, j, k] = 0.0

        elif mask[rolled_x, rolled_y] == 1:
            f_new[i, j, k] = f[i, j, reflection[k]]

        else:
            f_new[i, j, k] = f[rolled_x, rolled_y, k]

     # Store local momentum in shared memory
    if tid < shared_momentum.shape[0]:  # Ensure no out-of-bounds
        shared_momentum[tid] = local_momentum
    else:
        shared_momentum[tid] = 0.0
    cuda.syncthreads()

    # Perform reduction in shared memory
    stride = 1
    while stride < cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z:
        if tid % (2 * stride) == 0 and (tid + stride) < shared_momentum.shape[0]:
            shared_momentum[tid] += shared_momentum[tid + stride]
        stride *= 2
        cuda.syncthreads()

    # Write block-wise result to global memory
    if tid == 0:
        momentum_partial[block_id] = shared_momentum[0]

@cuda.jit
def global_reduce_kernel(momentum_partial, total_momentum):
    """Sum all block-wise results into a single value."""
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    shared_momentum = cuda.shared.array(1024, dtype=float64)

    # Load into shared memory
    if tid < len(momentum_partial):
        shared_momentum[cuda.threadIdx.x] = momentum_partial[tid]
    else:
        shared_momentum[cuda.threadIdx.x] = 0.0
    cuda.syncthreads()

    # Perform reduction in shared memory
    stride = 1
    while stride < cuda.blockDim.x:
        if cuda.threadIdx.x % (2 * stride) == 0:
            shared_momentum[cuda.threadIdx.x] += shared_momentum[cuda.threadIdx.x + stride]
        stride *= 2
        cuda.syncthreads()

    # Write result to global memory
    if cuda.threadIdx.x == 0:
        cuda.atomic.add(total_momentum, 0, shared_momentum[0])
