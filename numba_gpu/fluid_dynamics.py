# fluid_dynamics.py

from numba import cuda, float64


@cuda.jit(fastmath=True, cache=True)
def equilibrium_kernel(num_x, num_y, num_v, rho, u, feq, c, w, cs):
    """
    Evaluates the equilibrium distribution across the lattice for a
    given fluid density and velocity field

    Arguments:
        num_x (int): Lattice size in x-direction
        num_y (int): Lattice size in y-direction
        num_v (int): Number of velocity directions
        rho (array): 2D array of the fluid density at each lattice point
        u (array): 3D array of the fluid x & y velocity at each lattice point
        feq (array): Equilibrium distribution array
        c (array): Discrete velocity directions of shape (num_v, 2)
        w (array): Weight coefficients for velocity directions
        cs (float): Lattice speed of sound
    """

    # Pre-compute the speed of sound squared & to power of 4
    cs2 = cs * cs
    cs4 = cs2 * cs2

    i, j, k = cuda.grid(3) # Determine absolute position of current thread
    if i < num_x and j < num_y and k < num_v:
        u_dot_u = u[i, j, 0] * u[i, j, 0] + u[i, j, 1] * u[i, j, 1] # Magnitude squared of velocity
        u_dot_c = u[i, j, 0] * c[k, 0] + u[i, j, 1] * c[k, 1] # Velocity component in direction
        feq[i, j, k] = w[k] * (
                    1 + u_dot_c / cs2 +
                    (u_dot_c * u_dot_c) / (2 * cs4) -
                    u_dot_u / (2 * cs2)
                ) * rho[i, j]


@cuda.jit(fastmath=True, cache=True)
def fluid_density_kernel(num_x, num_y, num_v, f, rho, mask):
    """
    Calculate the fluid density from the distribution function.

    Arguments:
        num_x (int): Lattice size in x-direction
        num_y (int): Lattice size in y-direction
        num_v (int): Number of velocity directions
        f (array): Distribution function array
        rho (array): 2D array of the fluid density at each lattice point
        mask (array): Binary obstacle mask
    """

    i, j = cuda.grid(2) # Determine absolute position of current thread
    if i < num_x and j < num_y:
        if mask[i, j] == 1: # Set fluid density inside the obstacle
            rho[i, j] = 0.0001 # To avoid divisions by 0
        else:
            total = 0.0
            for k in range(num_v):
                total = total + f[i, j, k] # Sum over all velocity directions

            rho[i, j] = total


@cuda.jit(fastmath=True, cache=True)
def fluid_velocity_kernel(num_x, num_y, num_v, f, rho, u, c, mask):
    """
    Calculate the fluid velocity from the distribution function and fluid density.

    Arguments:
        num_x (int): Lattice size in x-direction
        num_y (int): Lattice size in y-direction
        num_v (int): Number of velocity directions
        f (array): Distribution function array
        rho (array): 2D array of the fluid density at each lattice point
        u (array): 3D array of the fluid x & y velocity at each lattice point
        c (array): Discrete velocity directions of shape (num_v, 2)
        mask (array): Binary obstacle mask
    """

    i, j = cuda.grid(2) # Determine absolute position of current thread
    if i < num_x and j < num_y:
        if mask[i, j] == 1:
            u[i, j, :] = 0 # Set velocity to 0 in the obstacle

        else:
            for k in range(num_v): # Sum contributions from all velocity directions
                u[i, j, 0] = u[i, j, 0] + (f[i, j, k] * c[k, 0] / rho[i, j])
                u[i, j, 1] = u[i, j, 1] + (f[i, j, k] * c[k, 1] / rho[i, j])


@cuda.jit(fastmath=True, cache=True)
def fluid_vorticity_kernel(u, vor):
    """
    Compute the vorticity of the velocity field.

    Arguments:
        u (array): 3D array of the fluid x & y velocity at each lattice point
        vor (array): 2D array of vorticity
    """

    i, j = cuda.grid(2) # Determine absolute position of current thread
    num_x, num_y = vor.shape

    if i < num_x and j < num_y:
        roll_up = (i - 1 + num_x) % num_x
        roll_down = (i + 1) % num_x
        roll_left = (j - 1 + num_y) % num_y
        roll_right = (j + 1) % num_y

        vor[i, j] = (u[roll_down, j, 1] - u[roll_up, j, 1] -
                     u[i, roll_right, 0] + u[i, roll_left, 0])


@cuda.jit(fastmath=True, cache=True)
def collision_kernel(num_x, num_y, num_v, f, feq, tau):
    """
    Perform the collision step, updating the distribution `f` using `feq`.

    Arguments:
        num_x (int): Lattice size in x-direction
        num_y (int): Lattice size in y-direction
        num_v (int): Number of velocity directions
        f (array): Distribution function array
        feq (array): Equilibrium distribution array
        tau (float): Decay timescale
    """

    i, j, k = cuda.grid(3) # Determine absolute position of current thread
    if i < num_x and j < num_y and k < num_v:
        f[i, j, k] = (f[i, j, k] * (1 - 1 / tau)) + (feq[i, j, k] / tau)


@cuda.jit(fastmath=True, cache=True)
def stream_and_reflect_kernel(num_x, num_y, num_v, f, f_new, momentum_point, u, mask, mask2, reflection, c, momentum_partial):
    """
    Perform the streaming and boundary reflection steps.

    Arguments:
        num_x (int): Lattice size in x-direction
        num_y (int): Lattice size in y-direction
        num_v (int): Number of velocity directions
        f (array): Distribution function array
        u (array): 3D array of the fluid x & y velocity at each lattice point
        c (array): Discrete velocity directions of shape (num_v, 2) 
        mask (array): Binary obstacle mask
        mask2 (array): Mask region used for force calculation
        reflection (array): Reflection mapping array
        momentum_point (array): Momentum array initialised as 0
        f_new (array): Streamed distribution function array initialised as 0
    """

    i, j, k = cuda.grid(3) # Determine absolute position of current thread

    # Shared memory for block-wise reduction
    shared_momentum = cuda.shared.array(1024, dtype=float64)  # Adjust size as needed

    # Thread ID within the block
    tid = cuda.threadIdx.x + cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.z * cuda.blockDim.x * cuda.blockDim.y
    block_id = cuda.blockIdx.x + cuda.blockIdx.y * cuda.gridDim.x + cuda.blockIdx.z * cuda.gridDim.x * cuda.gridDim.y

    local_momentum = 0.0

    # Calculate the source indices for streaming
    if i < num_x and j < num_y and k < num_v:
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
        local_momentum = local_momentum + momentum_point[i, j, k]

        # Perform streaming and reflection
        if mask[i, j] == 1:
            f_new[i, j, k] = 0.0 # No fluid inside obstacle

        elif mask[rolled_x, rolled_y] == 1:
            f_new[i, j, k] = f[i, j, reflection[k]] # Reflection

        else:
            f_new[i, j, k] = f[rolled_x, rolled_y, k] # Streaming

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


@cuda.jit(fastmath=True, cache=True)
def global_reduce_kernel(momentum_partial, total_momentum):
    """
    Sum all block-wise results into a single value for force calculation.

    Arguments:
        momentum_partial (array): Partial momentum contributions from threads
        total_momentum (array): Total transverse momentum at timestep
    """
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