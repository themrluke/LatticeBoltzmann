import os
import time
import numpy as np
from numba import cuda
import threading

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


def main():
    # Specify GPUs to use
    gpu_ids = [2]  # List of GPU IDs you want to use
    num_gpus = len(gpu_ids)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    print(f"Using GPUs: {gpu_ids}")

    # Initialize parameters
    sim = Parameters(num_x=3200, num_y=200, tau=0.7, u0=0.18, scalemax=0.015, t_steps=1000, t_plot=10)
    dvv_dir, streamlines_dir, test_streamlines_dir, test_mask_dir = setup_plot_directories()

    # Initialize density and velocity fields
    initial_rho, initial_u = initial_turbulence(sim)

    # Split simulation domain across GPUs
    x_chunks = np.array_split(np.arange(sim.num_x), num_gpus)

    # Set up CUDA grid and block dimensions for each chunk
    threads_per_block = (16, 4, 16)
    blocks_per_grid = [
        (
            (len(x_chunk) + threads_per_block[0] - 1) // threads_per_block[0],
            (sim.num_y + threads_per_block[1] - 1) // threads_per_block[1],
            (sim.num_v + threads_per_block[2] - 1) // threads_per_block[2],
        )
        for x_chunk in x_chunks
    ]

    threads_per_block_2d = (16, 4)
    blocks_per_grid_2d = [
        (
            (len(x_chunk) + threads_per_block_2d[0] - 1) // threads_per_block_2d[0],
            (sim.num_y + threads_per_block_2d[1] - 1) // threads_per_block_2d[1],
        )
        for x_chunk in x_chunks
    ]

    # Initialize results storage
    force_array = np.zeros(sim.t_steps)
    results = [None] * num_gpus

    # Define a function to run the simulation on a GPU thread
    def gpu_worker(gpu_id, x_chunk, grid_3d, grid_2d, result_index):
        cuda.select_device(gpu_id)
        results[result_index] = run_simulation_on_device(
            sim=sim,
            gpu_id = gpu_id,
            x_chunk=x_chunk,
            threads_per_block=threads_per_block,
            threads_per_block_2d=threads_per_block_2d,
            initial_rho_chunk=initial_rho[x_chunk],
            initial_u_chunk=initial_u[x_chunk],
            grid_3d=grid_3d,
            grid_2d=grid_2d,
            dvv_dir=dvv_dir,
            streamlines_dir=streamlines_dir,
            test_streamlines_dir=test_streamlines_dir,
            test_mask_dir=test_mask_dir,
            stream=cuda.stream()
        )

    # Launch GPU threads
    threads = []
    time_start = time.time()
    for gpu_id, (x_chunk, grid_3d, grid_2d, idx) in enumerate(zip(x_chunks, blocks_per_grid, blocks_per_grid_2d, range(num_gpus))):
        thread = threading.Thread(target=gpu_worker, args=(gpu_id, x_chunk, grid_3d, grid_2d, idx))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    time_end = time.time()
    print('TIME FOR TIMESTEP_LOOP FUNCTION: ', time_end - time_start)

    # Combine results from all GPUs
    for t in range(sim.t_steps):
        force_array[t] = sum(result[t] for result in results)

    # Save force data
    data_dir = 'Data'
    os.makedirs(data_dir, exist_ok=True)
    np.savetxt(os.path.join(data_dir, 'forces.csv'), force_array)


def run_simulation_on_device(sim, gpu_id, x_chunk, threads_per_block, threads_per_block_2d, initial_rho_chunk, initial_u_chunk, grid_3d, grid_2d, dvv_dir, streamlines_dir, test_streamlines_dir, test_mask_dir, stream):
    """Run the simulation on a single GPU."""
    # Allocate and initialize device arrays
    f_device = cuda.device_array((len(x_chunk), sim.num_y, sim.num_v), dtype=np.float64, stream=stream)
    feq_device = cuda.device_array_like(f_device)
    rho_device = cuda.device_array((len(x_chunk), sim.num_y), dtype=np.float64, stream=stream)
    u_device = cuda.device_array((len(x_chunk), sim.num_y, 2), dtype=np.float64, stream=stream)
    vor_device = cuda.device_array((len(x_chunk), sim.num_y), dtype=np.float64, stream=stream)
    f_new_device = cuda.device_array_like(f_device)
    momentum_point_device = cuda.device_array((len(x_chunk), sim.num_y, sim.num_v), dtype=np.float64, stream=stream)
    momentum_partial_device = cuda.device_array(len(x_chunk) * sim.num_y, dtype=np.float64, stream=stream)
    total_momentum_device = cuda.device_array(1, dtype=np.float64, stream=stream)

    # Copy constants to the GPU
    initial_rho_device = cuda.to_device(initial_rho_chunk, stream=stream)
    initial_u_device = cuda.to_device(initial_u_chunk, stream=stream)
    c_device = cuda.to_device(sim.c, stream=stream)
    w_device = cuda.to_device(sim.w, stream=stream)
    mask_device = cuda.to_device(sim.mask, stream=stream)
    mask2_device = cuda.to_device(sim.mask2, stream=stream)
    reflection_device = cuda.to_device(sim.reflection, stream=stream)

    equilibrium_kernel[grid_3d, threads_per_block, stream](
        initial_rho_device, initial_u_device, feq_device, c_device, w_device, sim.cs
    )
 
    fluid_density_kernel[grid_2d, threads_per_block_2d, stream](feq_device, rho_device, mask_device)
    
    u_device[:] = 0
    fluid_velocity_kernel[grid_2d, threads_per_block_2d, stream](feq_device, rho_device, u_device, c_device, mask_device)

    equilibrium_kernel[grid_3d, threads_per_block, stream](
        rho_device, u_device, feq_device, c_device, w_device, sim.cs
    )

    # For plotting or initialization, copy the results to the CPU
    if gpu_id == 0:
        fluid_vorticity_kernel[grid_2d, threads_per_block_2d, stream](u_device, vor_device)
        rho = rho_device.copy_to_host()
        u = u_device.copy_to_host()
        vor = vor_device.copy_to_host()

        plot_solution(sim, t=0, rho=rho, u=u, vor=vor,
                    dvv_dir=dvv_dir,
                    streamlines_dir=streamlines_dir, 
                    test_streamlines_dir=test_streamlines_dir,
                    test_mask_dir=test_mask_dir)

    # Time loop
    force_array = np.zeros(sim.t_steps)
    for t in range(1, sim.t_steps + 1):
        # Collision step
        collision_kernel[grid_3d, threads_per_block, stream](f_device, feq_device, sim.tau)
        
        # Streaming and reflection step
        stream_and_reflect_kernel[grid_3d, threads_per_block, stream](
            f_device, f_new_device, momentum_point_device, u_device, mask_device,
            mask2_device, reflection_device, c_device, momentum_partial_device
        )
        
        #Global reduction
        total_momentum_device[0] = 0.0  # Reset the total momentum accumulator
        global_reduce_kernel[grid_3d[0], threads_per_block[0], stream](
            momentum_partial_device, total_momentum_device
        )

        f_device, f_new_device = f_new_device, f_device # Swap buffers

        # Fluid properties updates
        fluid_density_kernel[grid_2d, threads_per_block_2d, stream](f_device, rho_device, mask_device)
        
        u_device[:] = 0
        fluid_velocity_kernel[grid_2d, threads_per_block_2d, stream](f_device, rho_device, u_device, c_device, mask_device)

        equilibrium_kernel[grid_3d, threads_per_block, stream](
            rho_device, u_device, feq_device, c_device, w_device, sim.cs
        )

        u_host = u_device.copy_to_host()
        feq_host = feq_device.copy_to_host()
        print(f"Step {t}: u max={np.max(u_host)}, min={np.min(u_host)}")
        print(f"Step {t}: feq max={np.max(feq_host)}, min={np.min(feq_host)}")
        
        # Optional vorticity calculation
        if t % sim.t_plot == 0 and gpu_id == 0:
            fluid_vorticity_kernel[grid_2d, threads_per_block_2d, stream](u_device, vor_device)
            # For plotting or initialization, copy the results to the CPU
            rho = rho_device.copy_to_host()
            u = u_device.copy_to_host()
            vor = vor_device.copy_to_host()

            plot_solution(sim, t=t, rho=rho, u=u, vor=vor,
                        dvv_dir=dvv_dir,
                        streamlines_dir=streamlines_dir, 
                        test_streamlines_dir=test_streamlines_dir,
                        test_mask_dir=test_mask_dir)
        
        force_array[t - 1] = total_momentum_device.copy_to_host(stream=stream)[0]

        f_device, f_new_device = f_new_device, f_device

    return force_array


if __name__ == "__main__":
    main()
