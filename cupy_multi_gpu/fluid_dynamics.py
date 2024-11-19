# fluid_dynamics.py

import cupy as cp

# Equilibrium kernel code
equilibrium_kernel_code = r'''
extern "C" __global__
void equilibrium_kernel(
    double* feq,
    const double* rho,
    const double* u,
    const double* c,
    const double* w,
    const double cs,
    const int num_x,
    const int num_y,
    const int num_v)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x-direction
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y-direction
    int k = blockIdx.z * blockDim.z + threadIdx.z; // velocity directions

    if (i < num_x && j < num_y && k < num_v) {
        int idx_f = (i * num_y + j) * num_v + k;
        int idx_rho = i * num_y + j;
        int idx_u = idx_rho * 2;

        double u_x = u[idx_u];
        double u_y = u[idx_u + 1];
        double c_x = c[k * 2];
        double c_y = c[k * 2 + 1];

        double u_dot_u = u_x * u_x + u_y * u_y;
        double u_dot_c = u_x * c_x + u_y * c_y;

        double feq_val = w[k] * (
            1 + u_dot_c / (cs * cs) +
            (u_dot_c * u_dot_c) / (2 * cs * cs * cs * cs) -
            u_dot_u / (2 * cs * cs)
        ) * rho[idx_rho];

        feq[idx_f] = feq_val;
    }
}
'''

# Compile the kernel
equilibrium_kernel = cp.RawKernel(equilibrium_kernel_code, 'equilibrium_kernel')

def equilibrium_kernel_launcher(**kwargs):
    equilibrium_kernel(
        (kwargs['grid']),
        (kwargs['block']),
        (kwargs['stream'].ptr if 'stream' in kwargs else 0),
        (
            kwargs['f_eq'].data.ptr,
            kwargs['rho'].data.ptr,
            kwargs['u'].data.ptr,
            kwargs['c'].data.ptr,
            kwargs['w'].data.ptr,
            kwargs['cs'],
            kwargs['num_x'],
            kwargs['num_y'],
            kwargs['num_v']
        )
    )

# Similarly, implement other kernels

# Collision kernel
collision_kernel_code = r'''
extern "C" __global__
void collision_kernel(
    double* f,
    const double* feq,
    const double tau,
    const int num_x,
    const int num_y,
    const int num_v)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x-direction
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y-direction
    int k = blockIdx.z * blockDim.z + threadIdx.z; // velocity directions

    if (i < num_x && j < num_y && k < num_v) {
        int idx = (i * num_y + j) * num_v + k;
        f[idx] = (1.0 - 1.0 / tau) * f[idx] + (1.0 / tau) * feq[idx];
    }
}
'''

collision_kernel = cp.RawKernel(collision_kernel_code, 'collision_kernel')

def collision_kernel_launcher(**kwargs):
    collision_kernel(
        (kwargs['grid']),
        (kwargs['block']),
        (kwargs['stream'].ptr if 'stream' in kwargs else 0),
        (
            kwargs['f'].data.ptr,
            kwargs['feq'].data.ptr,
            kwargs['tau'],
            kwargs['num_x'],
            kwargs['num_y'],
            kwargs['num_v']
        )
    )

# Streaming and reflection kernel
stream_and_reflect_kernel_code = r'''
extern "C" __global__
void stream_and_reflect_kernel(
    const double* f,
    double* f_new,
    const int* mask,
    const int* mask2,
    const int* reflection,
    const int* c,
    const int num_x,
    const int num_y,
    const int num_v)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x-direction
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y-direction
    int k = blockIdx.z * blockDim.z + threadIdx.z; // velocity directions

    if (i < num_x && j < num_y && k < num_v) {
        int idx = (i * num_y + j) * num_v + k;
        int c_x = c[k * 2];
        int c_y = c[k * 2 + 1];

        int rolled_i = (i - c_x + num_x) % num_x;
        int rolled_j = (j - c_y + num_y) % num_y;

        int idx_rolled = (rolled_i * num_y + rolled_j) * num_v + k;
        int idx_reflection = (i * num_y + j) * num_v + reflection[k];

        if (mask[idx / num_v] == 1) {
            f_new[idx] = 0.0;
        } else if (mask[rolled_i * num_y + rolled_j] == 1) {
            f_new[idx] = f[idx_reflection];
        } else {
            f_new[idx] = f[idx_rolled];
        }
    }
}
'''

stream_and_reflect_kernel = cp.RawKernel(stream_and_reflect_kernel_code, 'stream_and_reflect_kernel')

def stream_and_reflect_kernel_launcher(**kwargs):
    stream_and_reflect_kernel(
        (kwargs['grid']),
        (kwargs['block']),
        (kwargs['stream'].ptr if 'stream' in kwargs else 0),
        (
            kwargs['f'].data.ptr,
            kwargs['f_new'].data.ptr,
            kwargs['mask'].data.ptr,
            kwargs['mask2'].data.ptr,
            kwargs['reflection'].data.ptr,
            kwargs['c'].data.ptr,
            kwargs['num_x'],
            kwargs['num_y'],
            kwargs['num_v']
        )
    )

# Fluid density kernel
fluid_density_kernel_code = r'''
extern "C" __global__
void fluid_density_kernel(
    const double* f,
    double* rho,
    const int* mask,
    const int num_x,
    const int num_y,
    const int num_v)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x-direction
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y-direction

    if (i < num_x && j < num_y) {
        int idx_rho = i * num_y + j;

        if (mask[idx_rho] == 1) {
            rho[idx_rho] = 0.0001;
        } else {
            double total = 0.0;
            for (int k = 0; k < num_v; ++k) {
                int idx_f = idx_rho * num_v + k;
                total += f[idx_f];
            }
            rho[idx_rho] = total;
        }
    }
}
'''

fluid_density_kernel = cp.RawKernel(fluid_density_kernel_code, 'fluid_density_kernel')

def fluid_density_kernel_launcher(**kwargs):
    fluid_density_kernel(
        (kwargs['grid']),
        (kwargs['block']),
        (kwargs['stream'].ptr if 'stream' in kwargs else 0),
        (
            kwargs['f'].data.ptr,
            kwargs['rho'].data.ptr,
            kwargs['mask'].data.ptr,
            kwargs['num_x'],
            kwargs['num_y'],
            kwargs['num_v']
        )
    )

# fluid_dynamics.py (continuation)

# Fluid velocity kernel
fluid_velocity_kernel_code = r'''
extern "C" __global__
void fluid_velocity_kernel(
    const double* f,
    const double* rho,
    double* u,
    const double* c,
    const int* mask,
    const int num_x,
    const int num_y,
    const int num_v)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x-direction
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y-direction

    if (i < num_x && j < num_y) {
        int idx_rho = i * num_y + j;
        int idx_u = idx_rho * 2;

        if (mask[idx_rho] == 1) {
            u[idx_u] = 0.0;
            u[idx_u + 1] = 0.0;
        } else {
            double u_x = 0.0;
            double u_y = 0.0;
            double rho_val = rho[idx_rho];

            for (int k = 0; k < num_v; ++k) {
                int idx_f = idx_rho * num_v + k;
                double f_val = f[idx_f];
                double c_x = c[k * 2];
                double c_y = c[k * 2 + 1];
                u_x += f_val * c_x;
                u_y += f_val * c_y;
            }

            u[idx_u] = u_x / rho_val;
            u[idx_u + 1] = u_y / rho_val;
        }
    }
}
'''

fluid_velocity_kernel = cp.RawKernel(fluid_velocity_kernel_code, 'fluid_velocity_kernel')

def fluid_velocity_kernel_launcher(**kwargs):
    fluid_velocity_kernel(
        (kwargs['grid']),
        (kwargs['block']),
        (kwargs['stream'].ptr if 'stream' in kwargs else 0),
        (
            kwargs['f'].data.ptr,
            kwargs['rho'].data.ptr,
            kwargs['u'].data.ptr,
            kwargs['c'].data.ptr,
            kwargs['mask'].data.ptr,
            kwargs['num_x'],
            kwargs['num_y'],
            kwargs['num_v']
        )
    )


# Fluid vorticity kernel
fluid_vorticity_kernel_code = r'''
extern "C" __global__
void fluid_vorticity_kernel(
    const double* u,
    double* vor,
    const int num_x,
    const int num_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x-direction
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y-direction

    if (i < num_x && j < num_y) {
        int idx = i * num_y + j;

        // Compute indices with periodic boundary conditions
        int i_up = (i - 1 + num_x) % num_x;
        int i_down = (i + 1) % num_x;
        int j_left = (j - 1 + num_y) % num_y;
        int j_right = (j + 1) % num_y;

        int idx_up = i_up * num_y + j;
        int idx_down = i_down * num_y + j;
        int idx_left = i * num_y + j_left;
        int idx_right = i * num_y + j_right;

        // Velocity components
        double u_x_up = u[idx_up * 2];
        double u_x_down = u[idx_down * 2];
        double u_y_left = u[idx_left * 2 + 1];
        double u_y_right = u[idx_right * 2 + 1];

        // Vorticity calculation
        vor[idx] = (u_y_right - u_y_left) - (u_x_down - u_x_up);
    }
}
'''

fluid_vorticity_kernel = cp.RawKernel(fluid_vorticity_kernel_code, 'fluid_vorticity_kernel')

def fluid_vorticity_kernel_launcher(**kwargs):
    fluid_vorticity_kernel(
        (kwargs['grid']),
        (kwargs['block']),
        (kwargs['stream'].ptr if 'stream' in kwargs else 0),
        (
            kwargs['u'].data.ptr,
            kwargs['vor'].data.ptr,
            kwargs['num_x'],
            kwargs['num_y']
        )
    )


# Reduction kernel for computing partial sums
reduce_momentum_kernel_code = r'''
extern "C" __global__
void reduce_momentum_kernel(
    const double* data,
    double* result,
    const int n)
{
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (i < n) ? data[i] : 0.0;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result from each block to result array
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}
'''

reduce_momentum_kernel = cp.RawKernel(reduce_momentum_kernel_code, 'reduce_momentum_kernel')

def reduce_momentum_kernel_launcher(**kwargs):
    shared_mem_size = kwargs['block'][0] * cp.dtype(cp.float64).itemsize
    reduce_momentum_kernel(
        (kwargs['grid']),
        (kwargs['block']),
        (kwargs['stream'].ptr if 'stream' in kwargs else 0),
        (
            kwargs['data'].data.ptr,
            kwargs['result'].data.ptr,
            kwargs['n']
        ),
        shared_mem=shared_mem_size
    )

