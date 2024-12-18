# main.py

import argparse
import os 
import time 
import numpy as np
import numba
#import matplotlib.pyplot as plt
import cProfile
import pstats

from parameters import Parameters
from initialisation import InitialiseSimulation
from fluid_dynamics import equilibrium, collision, stream_and_reflect, fluid_density, fluid_velocity, fluid_vorticity
from plotting import plot_solution, setup_plot_directories


# Verify the threads
print(f"Using {numba.get_num_threads()} threads for Numba parallelization.")


def simulation_setup(num_x):
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
    sim = Parameters(num_x=num_x, num_y=200, tau=0.7, u0=0.18, scalemax=0.015, t_steps = 500, t_plot=100)

    # Initialise the simulation, obstacle and density & velocity fields
    initialiser = InitialiseSimulation(sim)
    initial_rho, initial_u = initialiser.initialise_turbulence(choice='m')

    # Set up plot directories
    directories = setup_plot_directories()

    # Allocate empty arrays once for arrays that need to be reset each timestep
    initialise_feq = np.zeros((sim.num_x, sim.num_y, sim.num_v), dtype=np.float64)
    initialise_rho = np.zeros((sim.num_x, sim.num_y), dtype=np.float64)
    initialise_u = np.zeros((sim.num_x, sim.num_y, 2), dtype=np.float64)
    initialise_momentum_point = np.zeros((sim.num_x, sim.num_y, sim.num_v), dtype=np.float64)
    initialise_f_new = np.zeros((sim.num_x, sim.num_y, sim.num_v), dtype=np.float64)
    reusable_arrays = (initialise_feq, initialise_rho, initialise_u, initialise_momentum_point, initialise_f_new)

    # Create the initial distribution by finding the equilibrium for the flow calculated above
    f = equilibrium(sim.num_x, sim.num_y, sim.num_v, initial_rho, initial_u, sim.c, sim.w, sim.cs, initialise_feq)
    rho = fluid_density(sim.num_x, sim.num_y, sim.num_v, f, sim.mask, initialise_rho)
    u = fluid_velocity(sim.num_x, sim.num_y, sim.num_v, f, rho, sim.c, sim.mask, initialise_u)
    feq = equilibrium(sim.num_x, sim.num_y, sim.num_v, rho, u, sim.c, sim.w, sim.cs, initialise_feq)

    return sim, rho, u, f, feq, reusable_arrays, directories


def timestep_loop(sim, rho, u, f, feq, reusable_arrays, directories):
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

    initialise_feq, initialise_rho, initialise_u, initialise_momentum_point, initialise_f_new = reusable_arrays

    force_array = np.zeros(sim.t_steps) # Initialising the array to store force values throughout simulation

    # Cache attributes that are repeatedly accessed
    num_x = sim.num_x
    num_y = sim.num_y
    num_v = sim.num_v
    tau = sim.tau
    c = sim.c
    mask = sim.mask
    mask2 = sim.mask2
    reflection = sim.reflection
    w = sim.w
    cs = sim.cs

    # Finally evolve the distribution in time
    time_start = time.time()
    for t in range(1, sim.t_steps + 1):

        # Perform collision step, using the calculated density and velocity data
        f = collision(num_x, num_y, num_v, f, feq, tau)

        # Streaming and reflection
        f, momentum_total = stream_and_reflect(num_x, num_y, num_v,
                                               f, u, c, mask, mask2, reflection,
                                               initialise_momentum_point, initialise_f_new)

        force_array[t-1] = momentum_total # Calculate the force at current timestep

        # Calculate density and velocity data, for next time around
        rho = fluid_density(num_x, num_y, num_v, f, mask, initialise_rho)
        u = fluid_velocity(num_x, num_y, num_v, f, rho, c, mask, initialise_u)

        # Recalculate equilibrium
        feq = equilibrium(num_x, num_y, num_v, rho, u, c, w, cs, initialise_feq)

        if (t % sim.t_plot == 0): # Visualise the simulation
            vor = fluid_vorticity(u)
            plot_solution(sim, t, rho, u, vor, *directories)
            print(f'PLOT {t} complete')

    time_end = time.time()
    execution_time = time_end - time_start
    print(f'TIME FOR TIMESTEP_LOOP FUNCTION: {execution_time}')

    # Append the result to a text file
    with open("loop_timings.txt", "a") as file:
        file.write(f"{execution_time}\n")

    return force_array


def main(num_x):

    # Setup simulation
    sim, rho, u, f, feq, reusable_arrays, directories = simulation_setup(num_x)

    # vor = fluid_vorticity(u)
    # plot_solution(sim, 0, rho, u, vor, *directories) # Visualise setup

    # Evolve simulation over time
    force_array = timestep_loop(sim, rho, u, f, feq, reusable_arrays, directories)

    # # Plot the force over time to make sure consistent between methods
    # plt.plot(np.arange(100, 1000, 1), np.asarray(force_array[100:]))
    # plt.savefig(f"plots/force_graph.png", dpi=300)
    # plt.close()

    # Save force data to CSV file
    data_dir = 'Data'
    os.makedirs(data_dir, exist_ok=True) # Ensure output directory exists
    np.savetxt(os.path.join(data_dir, 'forces.csv'), force_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulation with adjustable grid size.")
    parser.add_argument("--num_x", type=int, required=True, help="Number of grid points in the x direction.")
    args = parser.parse_args()

    profiler = cProfile.Profile()
    profiler.enable()
    main(args.num_x)
    profiler.disable()

    # Print the top 20 functions by cumulative time spent
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(20)