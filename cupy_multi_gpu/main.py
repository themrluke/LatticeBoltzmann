# main.py

import numpy as np
import os
import time

from parameters import Parameters
from initialisation import initial_turbulence
from fluid_dynamics import (
    equilibrium_kernel_launcher,
    collision_kernel_launcher,
    stream_and_reflect_kernel_launcher,
    fluid_density_kernel_launcher,
    fluid_velocity_kernel_launcher,
    fluid_vorticity_kernel_launcher,
)

from plotting import plot_solution, setup_plot_directories

import cupy as cp

def main():
    # Specify the GPUs you want to use
    gpu_ids = [4, 5]  # Use GPUs with IDs 0 and 2
    num_gpus = len(gpu_ids)
    print(f"Using GPUs: {gpu_ids}")

    # Validate GPU IDs
    available_gpus = cp.cuda.runtime.getDeviceCount()
    for gpu_id in gpu_ids:
        if gpu_id < 0 or gpu_id >= available_gpus:
            raise ValueError(f"GPU ID {gpu_id} is invalid. Available GPUs are from 0 to {available_gpus - 1}.")

    # Initialize parameters
    sim = Parameters(num_x=3200, num_y=200, tau=0.7, u0=0.18, scalemax=0.015, t_steps=500, t_plot=100)

    # Set up plot directories
    dvv_dir, streamlines_dir, test_streamlines_dir, test_mask_dir = setup_plot_directories()

    # Initialize density and velocity fields
    initial_rho, initial_u = initial_turbulence(sim)

    # Split the simulation domain across GPUs
    x_chunks = np.array_split(np.arange(sim.num_x), num_gpus)
    sim.num_x_chunks = [len(chunk) for chunk in x_chunks]

    # Prepare per-GPU data structures
    f_devices = []
    feq_devices = []
    rho_devices = []
    u_devices = []
    vor_devices = []
    f_new_devices = []
    streams = []

    # Prepare boundary buffers for inter-GPU communication
    left_boundary_buffers = []
    right_boundary_buffers = []

    # Constants (only need to be transferred once per GPU)
    c_device_list = []
    w_device_list = []
    mask_device_list = []
    mask2_device_list = []
    reflection_device_list = []

    for idx, gpu_id in enumerate(gpu_ids):
        with cp.cuda.Device(gpu_id):
            stream = cp.cuda.Stream()
            streams.append(stream)

            num_x_chunk = sim.num_x_chunks[idx]
            num_y = sim.num_y
            num_v = sim.num_v

            # Allocate arrays on this GPU
            f_device = cp.zeros((num_x_chunk, num_y, num_v), dtype=cp.float64)
            feq_device = cp.zeros_like(f_device)
            rho_device = cp.zeros((num_x_chunk, num_y), dtype=cp.float64)
            u_device = cp.zeros((num_x_chunk, num_y, 2), dtype=cp.float64)
            vor_device = cp.zeros((num_x_chunk, num_y), dtype=cp.float64)
            f_new_device = cp.zeros_like(f_device)

            f_devices.append(f_device)
            feq_devices.append(feq_device)
            rho_devices.append(rho_device)
            u_devices.append(u_device)
            vor_devices.append(vor_device)
            f_new_devices.append(f_new_device)

            # Allocate boundary buffers
            left_boundary = cp.zeros((num_y, num_v), dtype=cp.float64)
            right_boundary = cp.zeros((num_y, num_v), dtype=cp.float64)
            left_boundary_buffers.append(left_boundary)
            right_boundary_buffers.append(right_boundary)

            # Copy constants to the GPU
            c_device = cp.asarray(sim.c)
            w_device = cp.asarray(sim.w)
            mask_device = cp.asarray(sim.mask[x_chunks[idx], :])
            mask2_device = cp.asarray(sim.mask2[x_chunks[idx], :])
            reflection_device = cp.asarray(sim.reflection)

            c_device_list.append(c_device)
            w_device_list.append(w_device)
            mask_device_list.append(mask_device)
            mask2_device_list.append(mask2_device)
            reflection_device_list.append(reflection_device)

    # Distribute initial_rho and initial_u to GPUs and compute initial equilibrium
    for idx, gpu_id in enumerate(gpu_ids):
        with cp.cuda.Device(gpu_id):
            stream = streams[idx]

            x_chunk = x_chunks[idx]
            initial_rho_chunk = initial_rho[x_chunk]
            initial_u_chunk = initial_u[x_chunk]

            # Copy to device
            initial_rho_device = cp.asarray(initial_rho_chunk)
            initial_u_device = cp.asarray(initial_u_chunk)

            # Define grid dimensions
            num_x_chunk = sim.num_x_chunks[idx]
            num_y = sim.num_y
            num_v = sim.num_v

            threads_per_block = (16, 4, 4)
            blocks_per_grid = (
                (num_x_chunk + threads_per_block[0] - 1) // threads_per_block[0],
                (num_y + threads_per_block[1] - 1) // threads_per_block[1],
                (num_v + threads_per_block[2] - 1) // threads_per_block[2],
            )

            # Launch equilibrium kernel
            equilibrium_kernel_launcher(
                f_eq=feq_devices[idx],
                rho=initial_rho_device,
                u=initial_u_device,
                c=c_device_list[idx],
                w=w_device_list[idx],
                cs=sim.cs,
                num_x=num_x_chunk,
                num_y=num_y,
                num_v=num_v,
                block=threads_per_block,
                grid=blocks_per_grid,
                stream=stream,
            )

            # Initialize f_device
            f_devices[idx][:] = feq_devices[idx]

    # Main time-stepping loop
    time_start = time.time()
    for t in range(1, sim.t_steps + 1):
        # Collision and streaming steps
        for idx, gpu_id in enumerate(gpu_ids):
            with cp.cuda.Device(gpu_id):
                stream = streams[idx]

                num_x_chunk = sim.num_x_chunks[idx]
                num_y = sim.num_y
                num_v = sim.num_v

                threads_per_block = (16, 4, 4)
                blocks_per_grid = (
                    (num_x_chunk + threads_per_block[0] - 1) // threads_per_block[0],
                    (num_y + threads_per_block[1] - 1) // threads_per_block[1],
                    (num_v + threads_per_block[2] - 1) // threads_per_block[2],
                )

                # Collision kernel
                collision_kernel_launcher(
                    f=f_devices[idx],
                    feq=feq_devices[idx],
                    tau=sim.tau,
                    num_x=num_x_chunk,
                    num_y=num_y,
                    num_v=num_v,
                    block=threads_per_block,
                    grid=blocks_per_grid,
                    stream=stream,
                )

                # Streaming and reflection kernel
                stream_and_reflect_kernel_launcher(
                    f=f_devices[idx],
                    f_new=f_new_devices[idx],
                    mask=mask_device_list[idx],
                    mask2=mask2_device_list[idx],
                    reflection=reflection_device_list[idx],
                    c=c_device_list[idx],
                    num_x=num_x_chunk,
                    num_y=num_y,
                    num_v=num_v,
                    block=threads_per_block,
                    grid=blocks_per_grid,
                    stream=stream,
                )

        # Inter-GPU communication for periodic boundaries
        for idx, gpu_id in enumerate(gpu_ids):
            with cp.cuda.Device(gpu_id):
                stream = streams[idx]

                left_idx = (idx - 1) % num_gpus
                right_idx = (idx + 1) % num_gpus

                left_gpu_id = gpu_ids[left_idx]
                right_gpu_id = gpu_ids[right_idx]

                # Exchange boundary data asynchronously
                # Send right boundary to right neighbor's left buffer
                cp.cuda.runtime.memcpyPeerAsync(
                    dst=left_boundary_buffers[right_idx].data.ptr,
                    dstDevice=right_gpu_id,
                    src=f_new_devices[idx][-1, :, :].data.ptr,
                    srcDevice=gpu_id,
                    size=f_new_devices[idx][-1, :, :].nbytes,
                    stream=stream.ptr,
                )

                # Send left boundary to left neighbor's right buffer
                cp.cuda.runtime.memcpyPeerAsync(
                    dst=right_boundary_buffers[left_idx].data.ptr,
                    dstDevice=left_gpu_id,
                    src=f_new_devices[idx][0, :, :].data.ptr,
                    srcDevice=gpu_id,
                    size=f_new_devices[idx][0, :, :].nbytes,
                    stream=stream.ptr,
                )

        # Synchronize streams to ensure data transfer is complete
        for stream in streams:
            stream.synchronize()

        # Update boundary cells with received data
        for idx, gpu_id in enumerate(gpu_ids):
            with cp.cuda.Device(gpu_id):
                stream = streams[idx]

                # Update leftmost cell with data from left neighbor
                f_new_devices[idx][0, :, :] = right_boundary_buffers[idx]

                # Update rightmost cell with data from right neighbor
                f_new_devices[idx][-1, :, :] = left_boundary_buffers[idx]

        # Swap f and f_new
        for idx in range(num_gpus):
            f_devices[idx], f_new_devices[idx] = f_new_devices[idx], f_devices[idx]

        # Update fluid properties
        for idx, gpu_id in enumerate(gpu_ids):
            with cp.cuda.Device(gpu_id):
                stream = streams[idx]

                num_x_chunk = sim.num_x_chunks[idx]
                num_y = sim.num_y

                threads_per_block_2d = (16, 16)
                blocks_per_grid_2d = (
                    (num_x_chunk + threads_per_block_2d[0] - 1) // threads_per_block_2d[0],
                    (num_y + threads_per_block_2d[1] - 1) // threads_per_block_2d[1],
                )

                # Update density
                fluid_density_kernel_launcher(
                    f=f_devices[idx],
                    rho=rho_devices[idx],
                    mask=mask_device_list[idx],
                    num_x=num_x_chunk,
                    num_y=num_y,
                    num_v=sim.num_v,
                    block=threads_per_block_2d,
                    grid=blocks_per_grid_2d,
                    stream=stream,
                )

                # Update velocity
                fluid_velocity_kernel_launcher(
                    f=f_devices[idx],
                    rho=rho_devices[idx],
                    u=u_devices[idx],
                    c=c_device_list[idx],
                    mask=mask_device_list[idx],
                    num_x=num_x_chunk,
                    num_y=num_y,
                    num_v=sim.num_v,
                    block=threads_per_block_2d,
                    grid=blocks_per_grid_2d,
                    stream=stream,
                )

                # Compute equilibrium distribution
                threads_per_block = (16, 4, 4)
                blocks_per_grid = (
                    (num_x_chunk + threads_per_block[0] - 1) // threads_per_block[0],
                    (num_y + threads_per_block[1] - 1) // threads_per_block[1],
                    (sim.num_v + threads_per_block[2] - 1) // threads_per_block[2],
                )

                equilibrium_kernel_launcher(
                    f_eq=feq_devices[idx],
                    rho=rho_devices[idx],
                    u=u_devices[idx],
                    c=c_device_list[idx],
                    w=w_device_list[idx],
                    cs=sim.cs,
                    num_x=num_x_chunk,
                    num_y=num_y,
                    num_v=sim.num_v,
                    block=threads_per_block,
                    grid=blocks_per_grid,
                    stream=stream,
                )

        # Optional: Calculate vorticity and plotting
        if t % sim.t_plot == 0:
            # Gather data from GPUs
            rho_full = []
            u_full = []
            vor_full = []

            for idx, gpu_id in enumerate(gpu_ids):
                with cp.cuda.Device(gpu_id):
                    rho_chunk = rho_devices[idx].get()
                    u_chunk = u_devices[idx].get()

                    # Calculate vorticity
                    num_x_chunk = sim.num_x_chunks[idx]
                    num_y = sim.num_y

                    threads_per_block_2d = (16, 16)
                    blocks_per_grid_2d = (
                        (num_x_chunk + threads_per_block_2d[0] - 1) // threads_per_block_2d[0],
                        (num_y + threads_per_block_2d[1] - 1) // threads_per_block_2d[1],
                    )

                    fluid_vorticity_kernel_launcher(
                        u=u_devices[idx],
                        vor=vor_devices[idx],
                        num_x=num_x_chunk,
                        num_y=num_y,
                        block=threads_per_block_2d,
                        grid=blocks_per_grid_2d,
                        stream=streams[idx],
                    )

                    vor_chunk = vor_devices[idx].get()

                    rho_full.append(rho_chunk)
                    u_full.append(u_chunk)
                    vor_full.append(vor_chunk)

            # Concatenate the chunks
            rho = np.concatenate(rho_full, axis=0)
            u = np.concatenate(u_full, axis=0)
            vor = np.concatenate(vor_full, axis=0)

            # Plotting
            plot_solution(
                sim, t=t, rho=rho, u=u, vor=vor,
                dvv_dir=dvv_dir,
                streamlines_dir=streamlines_dir,
                test_streamlines_dir=test_streamlines_dir,
                test_mask_dir=test_mask_dir,
            )

    time_end = time.time()
    print('TIME FOR TIMESTEP_LOOP FUNCTION: ', time_end - time_start)

    # Save force data if applicable
    # ... (you can implement force calculation similar to the above pattern)

if __name__ == "__main__":
    main()
