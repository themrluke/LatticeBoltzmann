# main.py

from parameters import Parameters
from initialisation import initial_turbulence
from fluid_dynamics import timestep_loop
import matplotlib.pyplot as plt

import numpy as np
import os
import time

import cProfile
import pstats

def main():
    
    # Initialise parameters
    # CHANGE PARAMETER VALUES HERE.
    # Original parameters
    # num_x=3200, num_y=200, tau=0.500001, u0=0.18, scalemax=0.015, t_steps = 24000, t_plot=500
    sim = Parameters(num_x=3200, num_y=200, tau=0.7, u0=0.18, scalemax=0.015, t_steps = 500, t_plot=100)

    # Initialize density and velocity fields.
    initial_rho, initial_u = initial_turbulence(sim)

    # Finally evolve the distribution in time, using the 'collision' and
    # 'streaming_reflect' functions.
    time_start = time.time()
    force_array = timestep_loop(sim, initial_rho, initial_u)
    time_end = time.time()
    # plt.plot(np.arange(100, 1000, 1), np.asarray(force_array[100:]))
    # plt.savefig(f"plots/force_graph.png", dpi=300)
    # plt.close()
    print('TIME FOR TIMESTEP_LOOP FUNCTION: ', time_end - time_start)


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
    #profiler.dump_stats('profile_data.prof')  # Save the profile data to a file
    
    # Print the top 20 functions by cumulative time spent
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(20)