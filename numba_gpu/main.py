# main.py

import os
import time
import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
import cProfile
import pstats

from parameters import Parameters
from initialisation import InitialiseSimulation
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


# Verify available GPUs
print(f"CUDA devices: {cuda.gpus}")

device = cuda.get_current_device()
print("Threads per block:", device.MAX_THREADS_PER_BLOCK)
print("Threads per warp:", device.WARP_SIZE)
print("Shared memory per block:", device.MAX_SHARED_MEMORY_PER_BLOCK)
print("Registers per block:", device.MAX_REGISTERS_PER_BLOCK)

def simulation_setup():
    """
    Setup the Lattice Boltzmann parameters, initialise the obstacle and fields

    Returns:
        sim: Parameters object
        rho (np.ndarray): 2D array of the fluid density at each lattice point
        u (np.ndarray): 3D array of the fluid x & y velocity at each lattice point
        feq (np.ndarray): Equilibrium distribution array
        reusable_arrays (Tuple): Reusable arrays (initialise_feq, initialise_rho, initialise_u, initialise_momentum_point, initialise_f_new)
        directories (Tuple): Directories for different plot types
    """

    # Initialise parameters
    # num_x=3200, num_y=200, tau=0.500001, u0=0.18, scalemax=0.015, t_steps = 24000, t_plot=500
    sim = Parameters(num_x=3200, num_y=200, tau=0.7, u0=0.18, scalemax=0.015, t_steps = 500, t_plot=10000)

    # Initialise the simulation, obstacle and density & velocity fields
    initialiser = InitialiseSimulation(sim)
    initial_rho, initial_u = initialiser.initialise_turbulence(choice='m')

    # Set up plot directories
    directories = setup_plot_directories()

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
    initial_rho_device = cuda.to_device(initial_rho)
    initial_u_device = cuda.to_device(initial_u)
    c_device = cuda.to_device(sim.c)
    w_device = cuda.to_device(sim.w)
    mask_device = cuda.to_device(sim.mask)
    vor_device = cuda.device_array((sim.num_x, sim.num_y), dtype=np.float64)

    # Create the initial distribution by finding the equilibrium for the flow calculated above
    equilibrium_kernel[blocks_per_grid, threads_per_block](
        sim.num_x, sim.num_y, sim.num_v, initial_rho_device, initial_u_device, feq_device, c_device, w_device, sim.cs
    )
    fluid_density_kernel[blocks_per_grid_2d, threads_per_block_2d](
        sim.num_x, sim.num_y, sim.num_v, feq_device, rho_device, mask_device
    )
    u_device[:] = 0
    fluid_velocity_kernel[blocks_per_grid_2d, threads_per_block_2d](
        sim.num_x, sim.num_y, sim.num_v, feq_device, rho_device, u_device, c_device, mask_device
    )
    equilibrium_kernel[blocks_per_grid, threads_per_block](
        sim.num_x, sim.num_y, sim.num_v, rho_device, u_device, feq_device, c_device, w_device, sim.cs
    )

    return (
        sim, f_device, feq_device, rho_device, u_device, c_device, w_device, mask_device, vor_device, directories,
        threads_per_block, blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_v, blocks_per_grid,
        threads_per_block_2d, blocks_per_grid_2d
    )


def timestep_loop(sim, f_device, feq_device, rho_device, u_device, c_device, w_device, mask_device, vor_device, directories,
                  threads_per_block, blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_v, blocks_per_grid,
                  threads_per_block_2d, blocks_per_grid_2d
                  ):
    """
    Evolves the simulation over time

    Arguments:
        sim: Parameters object
        rho (np.ndarray): 2D array of the fluid density at each lattice point
        u (np.ndarray): 3D array of the fluid x & y velocity at each lattice point
        feq (np.ndarray): Equilibrium distribution array
        reusable_arrays (Tuple): Reusable arrays (initialise_feq, initialise_rho, initialise_u, initialise_momentum_point, initialise_f_new)
        directories (Tuple): Directories for different plot types

    Returns:
        force_array (np.ndarray): Total transverse force on obstacle for each timestep
    """
    
    # Preallocate arrays on the GPU
    f_new_device = cuda.device_array_like(f_device)
    momentum_point_device = cuda.device_array((sim.num_x, sim.num_y, sim.num_v), dtype=np.float64)
    momentum_partial_device = cuda.device_array(blocks_per_grid_x * blocks_per_grid_y * blocks_per_grid_v, dtype=np.float64)
    mask2_device = cuda.to_device(sim.mask2)
    reflection_device = cuda.to_device(sim.reflection)

    # Allocate global force accumulator
    total_momentum_device = cuda.device_array(1, dtype=np.float64)

    force_array = np.zeros(sim.t_steps) # Initialising the array to store force values throughout simulation

    # Cache attributes that are repeatedly accessed
    num_x = sim.num_x
    num_y = sim.num_y
    num_v = sim.num_v
    tau = sim.tau
    cs = sim.cs

    # Finally evolve the distribution in time
    time_start = time.time()
    for t in range(1, sim.t_steps + 1):

        # Perform collision step, using the calculated density and velocity data
        collision_kernel[blocks_per_grid, threads_per_block](
            num_x, num_y, num_v, f_device, feq_device, tau
        )

        # Streaming and reflection
        stream_and_reflect_kernel[blocks_per_grid, threads_per_block](
            num_x, num_y, num_v, f_device, f_new_device, momentum_point_device, u_device, mask_device, mask2_device, reflection_device, c_device, momentum_partial_device
        )

        total_momentum_device[0] = 0.0  # Reset the total momentum accumulator
        global_reduce_kernel[blocks_per_grid_x, threads_per_block[0]](
            momentum_partial_device, total_momentum_device
        )
        cuda.synchronize()

        force_array[t - 1] = total_momentum_device.copy_to_host()[0] # Calculate the force at current timestep

        # Swap buffers
        f_device, f_new_device = f_new_device, f_device

        # Calculate density and velocity data, for next time around
        fluid_density_kernel[blocks_per_grid_2d, threads_per_block_2d](
            num_x, num_y, num_v, f_device, rho_device, mask_device
        )
        u_device[:] = 0
        fluid_velocity_kernel[blocks_per_grid_2d, threads_per_block_2d](
            num_x, num_y, num_v, f_device, rho_device, u_device, c_device, mask_device
        )

        # Recalculate equilibrium
        equilibrium_kernel[blocks_per_grid, threads_per_block](
        num_x, num_y, num_v, rho_device, u_device, feq_device, c_device, w_device, cs
        )

        # if (t % sim.t_plot == 0): # Visualise the simulation
        #     fluid_vorticity_kernel[blocks_per_grid_2d, threads_per_block_2d](
        #         u_device, vor_device
        #     )
        #     rho = rho_device.copy_to_host()
        #     u = u_device.copy_to_host()
        #     vor = vor_device.copy_to_host()
        #     plot_solution(sim, t, rho, u, vor, *directories)
        #     print(f'PLOT {t} complete')

    time_end = time.time()
    print('TIME FOR TIMESTEP_LOOP FUNCTION: ', time_end - time_start)

    return force_array


def main():

    # Setup simulation
    (
    sim, f_device, feq_device, rho_device, u_device, c_device, w_device, mask_device, vor_device, directories,
    threads_per_block, blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_v, blocks_per_grid,
    threads_per_block_2d, blocks_per_grid_2d
    ) = simulation_setup()

    # # Visualise setup
    # fluid_vorticity_kernel[blocks_per_grid_2d, threads_per_block_2d](
    #     u_device, vor_device
    # )
    # rho = rho_device.copy_to_host()
    # u = u_device.copy_to_host()
    # vor = vor_device.copy_to_host()
    # plot_solution(sim, 0, rho, u, vor, *directories)

    # Evolve simulation over time
    force_array = timestep_loop(
        sim, f_device, feq_device, rho_device, u_device, c_device, w_device, mask_device, vor_device, directories,
        threads_per_block, blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_v, blocks_per_grid,
        threads_per_block_2d, blocks_per_grid_2d
        )

    # # Plot the force over time to make sure consistent between methods
    # plt.plot(np.arange(100, 1000, 1), np.asarray(force_array[100:]))
    # plt.savefig(f"plots/force_graph.png", dpi=300)
    # plt.close()

    # Save force data to CSV file
    data_dir = 'Data'
    os.makedirs(data_dir, exist_ok=True) # Ensure output directory exists
    np.savetxt(os.path.join(data_dir, 'forces.csv'), force_array)


if __name__ == "__main__":

    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()

    # Print the top 20 functions by cumulative time spent
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(20)