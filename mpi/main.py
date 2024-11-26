# main.py

import argparse
import os
import numpy as np
from mpi4py import MPI
#import matplotlib.pyplot as plt
import cProfile
import pstats

from parameters import Parameters
from initialisation import InitialiseSimulation
from fluid_dynamics import timestep_loop


def main(num_x):

    # Initialize MPI
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    # Initialise parameters
    # num_x=3200, num_y=200, tau=0.500001, u0=0.18, scalemax=0.015, t_steps = 24000, t_plot=500
    sim = Parameters(num_x=num_x, num_y=200, tau=0.7, u0=0.18, scalemax=0.015, t_steps = 500, t_plot=10000)

    # Divide the domain along the x-dimension
    local_num_x = sim.num_x // size
    if rank == size - 1:
        local_num_x += sim.num_x % size  # Last process takes the remainder rows

    local_start_x = rank * (sim.num_x // size)
    local_end_x = local_start_x + local_num_x

    # Initialise the simulation, obstacle and density & velocity fields
    initialiser = InitialiseSimulation(sim)
    initial_rho, initial_u = initialiser.initialise_turbulence(choice='m', start_x=local_start_x, end_x=local_end_x)

    # Evolve simulation over time
    local_force_array = timestep_loop(
        sim, initial_rho, initial_u, local_num_x, local_start_x, rank, size
    )

    global_force_array = np.empty((sim.t_steps), dtype=np.float64) # Initialise array to hold force data

    MPI.COMM_WORLD.Reduce(local_force_array, global_force_array, op=MPI.SUM, root=0)

    # if rank == 0:
    #     plt.plot(np.arange(100, 1000, 1), np.asarray(global_force_array[100:]))
    #     plt.savefig(f"plots/force_graph.png", dpi=300)
    #     plt.close()

    # Save force data to CSV file on rank 0
    if rank == 0:
        data_dir = "Data"
        os.makedirs(data_dir, exist_ok=True)
        np.savetxt(os.path.join(data_dir, "forces.csv"), global_force_array)
        print("Simulation completed")


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