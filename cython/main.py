# main.py

import argparse
import os
import numpy as np
#import matplotlib.pyplot as plt
import cProfile
import pstats

from parameters import Parameters
from initialisation import InitialiseSimulation
from fluid_dynamics import timestep_loop


def main(num_x):

    # Initialise parameters
    # num_x=3200, num_y=200, tau=0.500001, u0=0.18, scalemax=0.015, t_steps = 24000, t_plot=500
    sim = Parameters(num_x=num_x, num_y=200, tau=0.7, u0=0.18, scalemax=0.015, t_steps = 500, t_plot=10000)

    # Initialise the simulation, obstacle and density & velocity fields
    initialiser = InitialiseSimulation(sim)
    initial_rho, initial_u = initialiser.initialise_turbulence(choice='n')

    # Evolve the simulation over time
    force_array = timestep_loop(sim, initial_rho, initial_u)

    # plt.plot(np.arange(100, 1000, 1), np.asarray(force_array[100:]))
    # plt.savefig(f"plots/force_graph.png", dpi=300)
    # plt.close()

    # Save force data
    data_dir = 'Data'
    os.makedirs(data_dir, exist_ok=True) # Ensure output directory exists
    np.savetxt(os.path.join(data_dir, 'forces.csv'), force_array) # Save force data to CSV file in output dir


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