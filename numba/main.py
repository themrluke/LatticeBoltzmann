# main.py

from parameters import Parameters
from initialisation import initial_turbulence
from fluid_dynamics import equilibrium, collision, stream_and_reflect, fluid_density, fluid_velocity, fluid_vorticity
from plotting import plot_solution, setup_plot_directories

import numpy as np
import os
import time

import cProfile
import pstats


# Now import numba
import numba

# Verify the threads
print(f"Max available threads: {numba.config.NUMBA_DEFAULT_NUM_THREADS}")
numba.set_num_threads(8)
print(f"Using {numba.get_num_threads()} threads for Numba parallelization.")


def main():
    
    # Initialise parameters
    # CHANGE PARAMETER VALUES HERE.
    # Original parameters
    # num_x=3200, num_y=200, tau=0.500001, u0=0.18, scalemax=0.015, t_steps = 24000, t_plot=500
    sim = Parameters(num_x=3200, num_y=200, tau=0.7, u0=0.18, scalemax=0.015, t_steps = 500, t_plot=1000)

    # Set up plot directories
    dvv_dir, streamlines_dir, test_streamlines_dir, test_mask_dir = setup_plot_directories()

    # Initialize density and velocity fields.
    initial_rho, initial_u = initial_turbulence(sim)

    # Create the initial distribution by finding the equilibrium for the flow
    # calculated above.
    f = equilibrium(initial_rho, initial_u, sim.num_x, sim.num_y, sim.num_v, sim.c, sim.w, sim.cs)

    # We could just copy initial_rho, initial_v and f into rho, v and feq.
    rho = fluid_density(f, sim.num_x, sim.num_y, sim.num_v, sim.mask)
    u = fluid_velocity(f, rho, sim.num_x, sim.num_y, sim.num_v, sim.c, sim.mask)
    feq = equilibrium(rho, u, sim.num_x, sim.num_y, sim.num_v, sim.c, sim.w, sim.cs)
    vor = fluid_vorticity(u)

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
        print(f"Step {t} - f max: {np.max(f)}, f min: {np.min(f)}")
        print(f"Step {t} - u max: {np.max(u)}, u min: {np.min(u)}")

        # Perform collision step, using the calculated density and velocity data.
        time1_start = time.time()
        f = collision(f, feq, sim.num_x, sim.num_y, sim.num_v, sim.tau)
        time1_end = time.time()
        #print('collision() time: ', time1_end - time1_start)

        # Streaming and reflection
        time2_start = time.time()
        f, momentum_total = stream_and_reflect(f, u, sim.num_x, sim.num_y, sim.num_v,
                                               sim.c, sim.mask, sim.mask2, sim.reflection)
        time2_end = time.time()
        #print('stream_and_reflect() time: ', time2_end - time2_start)

        force_array[t-1] = momentum_total

        # Calculate density and velocity data, for next time around
        time3_start = time.time()
        rho = fluid_density(f, sim.num_x, sim.num_y, sim.num_v, sim.mask)
        time3_end = time.time()
        #print('fluid_density() time: ', time3_end - time3_start)

        time4_start = time.time()
        u = fluid_velocity(f, rho, sim.num_x, sim.num_y, sim.num_v, sim.c, sim.mask)
        time4_end = time.time()
        #print('fluid_velocity() time: ', time4_end - time4_start)

        time5_start = time.time()
        feq = equilibrium(rho, u, sim.num_x, sim.num_y, sim.num_v, sim.c, sim.w, sim.cs)
        time5_end = time.time()
        #print('equilibrium() time: ', time5_end - time5_start)
        
        if (t % sim.t_plot == 0):
            vor = fluid_vorticity(u)
            plot_solution(sim, t=t, rho=rho, u=u, vor=vor,
                          dvv_dir=dvv_dir,
                          streamlines_dir=streamlines_dir, 
                          test_streamlines_dir=test_streamlines_dir,
                          test_mask_dir=test_mask_dir,
                          )
    time_end = time.time()
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
    
    # Print the top 20 functions by cumulative time spent
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(20)