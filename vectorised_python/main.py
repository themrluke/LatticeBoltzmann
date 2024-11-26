# main.py

import argparse
import os
import time
import numpy as np
import cProfile
import pstats

from parameters import Parameters
from initialisation import InitialiseSimulation
from fluid_dynamics import equilibrium, collision, stream_and_reflect, fluid_density, fluid_velocity, fluid_vorticity
from plotting import plot_solution, setup_plot_directories


def simulation_setup(num_x):
    """
    Setup the Lattice Boltzmann parameters, initialise the obstacle and fields

    Returns:
        sim: Parameters object
        rho (np.ndarray): 2D array of the fluid density at each lattice point
        u (np.ndarray): 3D array of the fluid x & y velocity at each lattice point
        feq (np.ndarray): Equilibrium distribution array
        directories (Tuple): Directories for different plot types
    """

    # Initialise parameters
    # num_x=3200, num_y=200, tau=0.500001, u0=0.18, scalemax=0.015, t_steps = 24000, t_plot=500
    sim = Parameters(num_x=num_x, num_y=200, tau=0.7, u0=0.18, scalemax=0.015, t_steps = 500, t_plot=1000)

    # Initialise the simulation, obstacle and density & velocity fields
    initialiser = InitialiseSimulation(sim)
    initial_rho, initial_u = initialiser.initialise_turbulence(choice='n')

    # Set up plot directories
    directories = setup_plot_directories()

    # Create the initial distribution by finding the equilibrium for the flow calculated above
    f = equilibrium(sim, initial_rho, initial_u)
    rho = fluid_density(sim, f)
    u = fluid_velocity(sim, f, rho)
    feq = equilibrium(sim, rho, u)

    return sim, rho, u, f, feq, directories


def timestep_loop(sim, rho, u, f, feq, directories):
    """
    Evolves the simulation over time

    Arguments:
        sim: Parameters object
        rho (np.ndarray): 2D array of the fluid density at each lattice point
        u (np.ndarray): 3D array of the fluid x & y velocity at each lattice point
        feq (np.ndarray): Equilibrium distribution array
        directories (Tuple): Directories for different plot types

    Returns:
        force_array (np.ndarray): Total transverse force on obstacle for each timestep
    """

    force_array = np.zeros((sim.t_steps)) # Initialising the array to store force values throughout simulation

    # Finally evolve the distribution in time
    time_start = time.time()
    for t in range(1, sim.t_steps + 1):

        # Perform collision step, using the calculated density and velocity data
        f = collision(sim, f, feq)

        # Streaming and reflection
        f, momentum_total = stream_and_reflect(sim, f, u)

        force_array[t-1] = momentum_total # Calculate the force at current timestep

        # Calculate density and velocity data, for next time around
        rho = fluid_density(sim, f)
        u = fluid_velocity(sim, f, rho)

        # Recalculate equilibrium
        feq = equilibrium(sim, rho, u)

        # if (t % sim.t_plot == 0): # Visualise the simulation
        #     vor = fluid_vorticity(u)
        #     plot_solution(sim, t, rho, u, vor, *directories)

    time_end = time.time()
    execution_time = time_end - time_start
    print(f'TIME FOR TIMESTEP_LOOP FUNCTION: {execution_time}')

    # Append the result to a text file
    with open("loop_timings.txt", "a") as file:
        file.write(f"{execution_time}\n")

    return force_array


def main(num_x):

    # Setup simulation
    sim, rho, u, f, feq, directories = simulation_setup(num_x)

    # vor = fluid_vorticity(u)
    # plot_solution(sim, 0, rho, u, vor, *directories) # Visualise setup

    # Evolve simulation over time
    force_array = timestep_loop(sim, rho, u, f, feq, directories)

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