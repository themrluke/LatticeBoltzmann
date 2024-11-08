# main_single_thread.py
from parameters import Parameters
from initialisation import initial_turbulence
from fluid_dynamics import equilibrium, collision, stream_and_reflect, fluid_density, fluid_velocity, fluid_vorticity
from plotting import plot_solution, setup_plot_directories

import numpy as np
import time
import os

import cProfile
import pstats

def main():
    
    # Initialise parameters
    # CHANGE PARAMETER VALUES HERE.
    # Original parameters
    # num_x=3200, num_y=200, tau=0.500001, u0=0.18, scalemax=0.015, t_steps = 24000, t_plot=500
    sim = Parameters(num_x=3200, num_y=200, tau=0.7, u0=0.18, scalemax=0.015, t_steps = 10, t_plot=100)

    # Set up plot directories
    dvv_dir, streamlines_dir, test_streamlines_dir, test_mask_dir = setup_plot_directories()

    # Initialize density and velocity fields.
    initial_rho, initial_u = initial_turbulence(sim)

    # Create the initial distribution by finding the equilibrium for the flow
    # calculated above.
    f = equilibrium(sim, initial_rho, initial_u)

    # We could just copy initial_rho, initial_v and f into rho, v and feq.
    rho = fluid_density(sim, f)
    u = fluid_velocity(sim, f, rho)
    feq = equilibrium(sim, rho, u)
    vor = fluid_vorticity(sim, u)

    # plot_solution(sim, t=0, rho=rho, u=u, vor=vor,
    #               dvv_dir=dvv_dir,
    #               streamlines_dir=streamlines_dir, 
    #               test_streamlines_dir=test_streamlines_dir,
    #               test_mask_dir=test_mask_dir,
    #               )

    # Finally evolve the distribution in time, using the 'collision' and
    # 'streaming_reflect' functions.
    force_array = np.zeros((sim.t_steps)) #initialising the array which will store the force throughout the whole simulation
    for t in range(1, sim.t_steps + 1):
        print(f"Step {t} - f max: {np.max(f)}, f min: {np.min(f)}")
        print(f"Step {t} - u max: {np.max(u)}, u min: {np.min(u)}")

        # Perform collision step, using the calculated density and velocity data.
        f = collision(sim, f, feq)

        # Streaming and reflection
        f, momentum_total = stream_and_reflect(sim, f, u)
        force_array[t-1] = momentum_total
        # Calculate density and velocity data, for next time around
        rho = fluid_density(sim, f)
        u = fluid_velocity(sim, f, rho)
        feq = equilibrium(sim, rho, u)
        #print('reynolds number: ', sim.Re)
        
        # if (t % sim.t_plot == 0):
        #     vor = fluid_vorticity(sim, u)
        #     plot_solution(sim, t=t, rho=rho, u=u, vor=vor,
        #                   dvv_dir=dvv_dir,
        #                   streamlines_dir=streamlines_dir, 
        #                   test_streamlines_dir=test_streamlines_dir,
        #                   test_mask_dir=test_mask_dir,
        #                   )

    
    data_dir = 'Data'
    os.makedirs(data_dir, exist_ok=True) # Ensure output directory exists
    np.savetxt(os.path.join(data_dir, 'forces.csv'), force_array) # Save force data to CSV file in output dir
    # edit each time file creation names here and in plot_solution() function   

# Run the main function if the script is executed directly
if __name__ == "__main__":

    profiler = cProfile.Profile()
    profiler.enable()
    main_time_start = time.time()
    main()
    main_time_end = time.time()
    print('TIME FOR PROGRAM: ', main_time_end - main_time_start)
    profiler.disable()
    #profiler.dump_stats('profile_data.prof')  # Save the profile data to a file
    
    # Print the top 20 functions by cumulative time spent
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(20)