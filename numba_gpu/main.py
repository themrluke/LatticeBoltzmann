# main.py

from parameters import Parameters
from initialisation import initial_turbulence
from fluid_dynamics import (
    equilibrium_kernel,
    collision_kernel,
    stream_and_reflect_kernel,
    global_reduce_kernel,
    fluid_density_kernel,
    fluid_velocity_kernel,
    fluid_vorticity_kernel,
)

from plotting import plot_solution, setup_plot_directories

import numpy as np
import os
import time

import cProfile
import pstats

from numba import cuda
import matplotlib.pyplot as plt

# Verify the threads
print(f"CUDA devices: {cuda.gpus}")

device = cuda.get_current_device()
print("Threads per block:", device.MAX_THREADS_PER_BLOCK)
print("Threads per warp:", device.WARP_SIZE)
print("Shared memory per block:", device.MAX_SHARED_MEMORY_PER_BLOCK)
print("Registers per block:", device.MAX_REGISTERS_PER_BLOCK)


def main():
    
    # Initialise parameters
    # CHANGE PARAMETER VALUES HERE.
    # Original parameters
    # num_x=3200, num_y=200, tau=0.500001, u0=0.18, scalemax=0.015, t_steps = 24000, t_plot=500
    sim = Parameters(num_x=3200, num_y=200, tau=0.7, u0=0.18, scalemax=0.015, t_steps = 500, t_plot=100)
    
    # Set up plot directories
    dvv_dir, streamlines_dir, test_streamlines_dir, test_mask_dir = setup_plot_directories()

    # Initialize density and velocity fields.
    initial_rho, initial_u = initial_turbulence(sim)

    # CUDA grid and block dimensions
    threads_per_block = (16, 4, 16) # x*y*z should be a multiple of 32
    blocks_per_grid_x = (sim.num_x + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (sim.num_y + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid_v = (sim.num_v + threads_per_block[2] - 1) // threads_per_block[2]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_v)

    threads_per_block_2d = (16, 4)
    blocks_per_grid_2d_x = (sim.num_x + threads_per_block_2d[0] - 1) // threads_per_block_2d[0]
    blocks_per_grid_2d_y = (sim.num_y + threads_per_block_2d[1] - 1) // threads_per_block_2d[1]
    blocks_per_grid_2d = (blocks_per_grid_2d_x, blocks_per_grid_2d_y)

    # Preallocate arrays on the GPU
    f_device = cuda.device_array((sim.num_x, sim.num_y, sim.num_v), dtype=np.float64)
    feq_device = cuda.device_array_like(f_device)
    rho_device = cuda.device_array((sim.num_x, sim.num_y), dtype=np.float64)
    u_device = cuda.device_array((sim.num_x, sim.num_y, 2), dtype=np.float64)
    vor_device = cuda.device_array((sim.num_x, sim.num_y), dtype=np.float64)
    f_new_device = cuda.device_array_like(f_device)
    momentum_point_device = cuda.device_array((sim.num_x, sim.num_y, sim.num_v), dtype=np.float64)
    momentum_partial_device = cuda.device_array(blocks_per_grid_x * blocks_per_grid_y * blocks_per_grid_v, dtype=np.float64)
    
    # Allocate global force accumulator
    total_momentum_device = cuda.device_array(1, dtype=np.float64)

    # Copy constants to the GPU
    initial_rho_device = cuda.to_device(initial_rho)
    initial_u_device = cuda.to_device(initial_u)
    c_device = cuda.to_device(sim.c)
    w_device = cuda.to_device(sim.w)
    mask_device = cuda.to_device(sim.mask)
    mask2_device = cuda.to_device(sim.mask2)
    reflection_device = cuda.to_device(sim.reflection)

    equilibrium_kernel[blocks_per_grid, threads_per_block](
        initial_rho_device,
        initial_u_device,
        feq_device,
        c_device,
        w_device,
        sim.cs,
    )
    #cuda.synchronize()

    # Calculate initial fluid density
    fluid_density_kernel[blocks_per_grid_2d, threads_per_block_2d](feq_device, rho_device, mask_device)
    #cuda.synchronize()

    # Calculate initial fluid velocity
    u_device[:] = 0
    fluid_velocity_kernel[blocks_per_grid_2d, threads_per_block_2d](feq_device, rho_device, u_device, c_device, mask_device)
    #cuda.synchronize()

    # Calculate initial equilibrium distribution
    equilibrium_kernel[blocks_per_grid, threads_per_block](
        rho_device, u_device, feq_device, c_device, w_device, sim.cs
    )
    #cuda.synchronize()

    # Optional: Calculate initial vorticity for plotting
    fluid_vorticity_kernel[blocks_per_grid_2d, threads_per_block_2d](u_device, vor_device)
    #cuda.synchronize()

    # For plotting or initialization, copy the results to the CPU
    rho = rho_device.copy_to_host()
    u = u_device.copy_to_host()
    vor = vor_device.copy_to_host()

    plot_solution(sim, t=0, rho=rho, u=u, vor=vor,
                  dvv_dir=dvv_dir,
                  streamlines_dir=streamlines_dir, 
                  test_streamlines_dir=test_streamlines_dir,
                  test_mask_dir=test_mask_dir)

    # Finally evolve the distribution in time, using the 'collision' and
    # 'streaming_reflect' functions.
    force_array = np.zeros((sim.t_steps)) #initialising the array which will store the force throughout the whole simulation

    time_start = time.time()
    for t in range(1, sim.t_steps + 1):

        # Collision step
        collision_kernel[blocks_per_grid, threads_per_block](f_device, feq_device, sim.tau)
        #cuda.synchronize()

        # Streaming and reflection step
        stream_and_reflect_kernel[blocks_per_grid, threads_per_block](
            f_device,
            f_new_device,
            momentum_point_device,
            u_device,
            mask_device,
            mask2_device,
            reflection_device,
            c_device,
            momentum_partial_device
        )
        
        
        total_momentum_device[0] = 0.0  # Reset the total momentum accumulator
        global_reduce_kernel[blocks_per_grid_x, threads_per_block[0]](momentum_partial_device, total_momentum_device)
        cuda.synchronize()

        total_momentum_host = total_momentum_device.copy_to_host()
        force_array[t - 1] = total_momentum_host[0]

        # u_host = u_device.copy_to_host()
        # feq_host = feq_device.copy_to_host()
        # print(f"Step {t}: u max={np.max(u_host)}, min={np.min(u_host)}")
        # print(f"Step {t}: feq max={np.max(feq_host)}, min={np.min(feq_host)}")

        # Swap buffers
        f_device, f_new_device = f_new_device, f_device

        # Update fluid density
        fluid_density_kernel[blocks_per_grid_2d, threads_per_block_2d](f_device, rho_device, mask_device)
        #cuda.synchronize()

        # Update fluid velocity
        u_device[:] = 0
        fluid_velocity_kernel[blocks_per_grid_2d, threads_per_block_2d](f_device, rho_device, u_device, c_device, mask_device)
        #cuda.synchronize()

        equilibrium_kernel[blocks_per_grid, threads_per_block](
        rho_device, u_device, feq_device, c_device, w_device, sim.cs
        )
        #cuda.synchronize()

        #Calculate vorticity (optional for plotting)
        if t % sim.t_plot == 0:
            fluid_vorticity_kernel[blocks_per_grid_2d, threads_per_block_2d](u_device, vor_device)
            #cuda.synchronize()

            # Copy data back for plotting
            rho = rho_device.copy_to_host()
            u = u_device.copy_to_host()
            vor = vor_device.copy_to_host()
            plot_solution(sim, t=t, rho=rho, u=u, vor=vor, dvv_dir=dvv_dir,
                          streamlines_dir=streamlines_dir,
                          test_streamlines_dir=test_streamlines_dir,
                          test_mask_dir=test_mask_dir)
    time_end = time.time()
    print('TIME FOR TIMESTEP_LOOP FUNCTION: ', time_end - time_start)


    # plt.plot(np.arange(100, 1000, 1), np.asarray(force_array[100:]))
    # plt.savefig(f"plots/force_graph.png", dpi=300)
    # plt.close()

    data_dir = 'Data'
    os.makedirs(data_dir, exist_ok=True) # Ensure output directory exists
    np.savetxt(os.path.join(data_dir, 'forces.csv'), force_array) # Save force data to CSV file in output dir
    # edit each time file creation names here and in plot_solution() function   

# Run the main function if the script is executed directly
if __name__ == "__main__":

    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    
    # Print the top 20 functions by cumulative time spent
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(20)