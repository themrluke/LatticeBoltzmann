# main.py
import numpy as np
from mpi4py import MPI
import os
import time

from parameters import Parameters
from initialisation import initial_turbulence
from fluid_dynamics import equilibrium, fluid_density, fluid_velocity, timestep_loop
#from plotting import plot_solution, setup_plot_directories

import cProfile
import pstats

def main():

    # Initialize MPI
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    # Initialise parameters
    # CHANGE PARAMETER VALUES HERE.
    # Original parameters
    # num_x=3200, num_y=200, tau=0.500001, u0=0.18, scalemax=0.015, t_steps = 24000, t_plot=500
    sim = Parameters(num_x=3200, num_y=200, tau=0.7, u0=0.18, scalemax=0.015, t_steps = 10, t_plot=100)
    sim = MPI.COMM_WORLD.bcast(sim if rank == 0 else None, root=0)

    # Set up plot directories
    #dvv_dir, streamlines_dir, test_streamlines_dir, test_mask_dir = setup_plot_directories()

    # Divide the domain along the x-dimension
    local_num_x = sim.num_x // size
    if rank == size - 1:
        local_num_x += sim.num_x % size  # Last process takes the remainder rows

    local_start_x = rank * (local_num_x)
    local_end_x = local_start_x + local_num_x

    # Initialize density and velocity fields.
    initial_rho, initial_u = initial_turbulence(sim, local_start_x, local_end_x)

    # Create arrays for boundaries
    left_ghost = np.empty((sim.num_y, sim.num_v), dtype=np.float64)
    right_ghost = np.empty((sim.num_y, sim.num_v), dtype=np.float64)

    #vor = fluid_vorticity(sim, u)

    # plot_solution(sim, t=0, rho=rho, u=u, vor=vor,
    #               dvv_dir=dvv_dir,
    #               streamlines_dir=streamlines_dir, 
    #               test_streamlines_dir=test_streamlines_dir,
    #               test_mask_dir=test_mask_dir,
    #               )

    # Finally evolve the distribution in time, using the 'collision' and
    # 'streaming_reflect' functions.
    time_start = time.time()
    local_force_array = timestep_loop(sim,
                                      initial_rho,
                                      initial_u,
                                      local_num_x,
                                      local_start_x,
                                      local_end_x,
                                      rank,
                                      size)
    time_end = time.time()
    print('TIME FOR TIMESTEP_LOOP FUNCTION: ', time_end - time_start)

    # Gather force arrays from all processes
    if rank == 0:
        global_force_array = np.empty((sim.t_steps * size), dtype=np.float64)
    else:
        global_force_array = None

    MPI.COMM_WORLD.Gather(local_force_array, global_force_array, root=0)

    # Save results on rank 0
    if rank == 0:
        data_dir = "Data"
        os.makedirs(data_dir, exist_ok=True)
        np.savetxt(os.path.join(data_dir, "forces.csv"), global_force_array)
        print("Simulation completed")

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