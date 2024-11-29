# plotting.py

import matplotlib.pyplot as plt
import numpy as np
import os
import gc


plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def setup_plot_directories(plot_dir='plots'):
    """
    Create directories for different types of plots

    Arguments:
        plot_dir (str): Base directory for saving plots

    Returns:
        dvv_dir (str): Directory to save density, velocity, vorticity plots
        streamlines_dir (str): Directory to save vorticity with streamlines plots
        test_streamlines_dir (str): Directory to save test vorticity with streamlines plots without cutoff
        test_mask_dir (str): Directory to save mask2 visualisation plot for testing
    """

    os.makedirs(plot_dir, exist_ok=True)

    # Define subdirectories for each type of plot
    dvv_dir = os.path.join(plot_dir, "DVV")
    streamlines_dir = os.path.join(plot_dir, "streamlines")
    test_streamlines_dir = os.path.join(plot_dir, "Test_streamlines")
    test_mask_dir = os.path.join(plot_dir, "Test_mask")

    # Create subdirectories if they do not exist
    os.makedirs(dvv_dir, exist_ok=True)
    os.makedirs(streamlines_dir, exist_ok=True)
    os.makedirs(test_streamlines_dir, exist_ok=True)
    os.makedirs(test_mask_dir, exist_ok=True)

    return dvv_dir, streamlines_dir, test_streamlines_dir, test_mask_dir


def plot_solution(sim, t, rho, u, vor, dvv_dir, streamlines_dir, test_streamlines_dir, test_mask_dir):
    """
    Create and save plots for density, velocity and vorticity fields,
    as well as diagnostic plots for debugging

    Arguments:
        sim: Simulation parameters object
        t (int): current timestep
        rho (np.ndarray): 2D array of the fluid density at each lattice point
        u (np.ndarray): 3D array of the fluid x & y velocity at each lattice point
        vor (np.ndarray): 2D array of vorticity
        dvv_dir (str): Directory to save density, velocity, vorticity plots
        streamlines_dir (str): Directory to save vorticity with streamlines plots
        test_streamlines_dir (str): Directory to save test vorticity with streamlines plots without cutoff
        test_mask_dir (str): Directory to save mask2 visualisation plot for testing
    """

    # We have naturally used x as the first index in our multidimensional
    # arrays and y as the second index. However, matplotlib essentially plots
    # *matrices*, with first index increasing *down* the matrix and the second
    # index increasing *across*. This explains the use of 'transpose' and
    # *origin = 'lower'* in the code below.

    # Center on grid points
    x = np.arange(sim.num_x) + 0.5
    y = np.arange(sim.num_y) + 0.5

    # Needed to make the length of system longer to avoid interference from voertices because of periodic boundaries
    # Cutoff value allows the graph to remain focused on the interesting part of the vortex near the cars
    # Cutoff limits how far the graphs are plotted along x-direction. 
    # Cutoff = sim.num_x for true scale of simulated fluid
    cutoff = 1300



    [fig, ax] = plt.subplots(3, 1, figsize=(16,9))

    # DENSITY FIELD PLOT
    c = ax[0].imshow(rho[:cutoff, :].transpose(), origin='lower', extent=[0,cutoff,0,sim.num_y], vmax=1.2) 
    bar = fig.colorbar(c,ax=ax[0])
    bar.set_label(r'$\rho$')
    ax[0].set_title(r'Density $\rho$')
    ax[0].set_xlabel('$X$')
    ax[0].set_ylabel('$Y$')

    # VELOCITY FIELD PLOT
    speed = np.sqrt(np.einsum('xyv->xy', u*u) ) / sim.u0
    c = ax[1].imshow(speed[:cutoff, :].transpose(), origin='lower', extent=[0,cutoff,0,sim.num_y],
                       vmin=0, vmax=1)
    bar = fig.colorbar(c, ax=ax[1])
    bar.set_label(r'$|\mathbf{u}|/u_0$')
    ax[1].streamplot(x[:cutoff], y, u[:cutoff,:,0].transpose(),
                       u[:cutoff,:,1].transpose(), color=[1,1,1], density = 1, linewidth = 0.7, arrowsize = 0.7)
    ax[1].set_title(r'Velocity $\mathbf{u}$')
    ax[1].set_xlabel('$X$')
    ax[1].set_ylabel('$Y$')

    # VORTICITY FIELD PLOT
    c = ax[2].imshow(vor[:cutoff, :].transpose(), origin='lower', extent=[0,cutoff,0,sim.num_y])
    bar = fig.colorbar(c,ax=ax[2])
    bar.set_label('$v$')
    ax[2].set_title(r'Vorticity $v$')
    ax[2].set_xlabel('$X$')
    ax[2].set_ylabel('$Y$')

    plt.subplots_adjust(hspace=0.85)
    plt.savefig(f"{dvv_dir}/DVV_{t}.png", dpi=300)
    plt.close()



    # VORTICITY FIELD WITH STREAMLINES PLOT
    # Colourmaps that work well: twilight, terrain, nipy_spectral, gist_ncar
    c = plt.imshow(vor[:cutoff, :].transpose(), origin='lower', extent=[0,cutoff,0,sim.num_y], cmap='gist_ncar', vmin=sim.scalemin, vmax=sim.scalemax)
    #bar2 = plt.colorbar(c)
    #bar2.set_label('$v$')
    plt.title(r'Vorticity with Streamlines $v$', fontsize = '8')
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    #plt.streamplot(x[:cutoff], y, u[:cutoff, :, 0].transpose(), u[:cutoff,:,1].transpose(), color=[1,1,1], density = 0.87, linewidth = 0.4, arrowsize = 0.4)
    plt.tight_layout()
    plt.savefig(f"{streamlines_dir}/streamlines_{t}.png", dpi=300)
    plt.close()



    # TEST VORTICITY PLOT
    # To check if vortices are interfering with front of car due to going off graph at back
    c = plt.imshow(vor.transpose(), origin='lower', extent=[0,sim.num_x,0,sim.num_y], cmap='gist_ncar', vmin=sim.scalemin, vmax=sim.scalemax)
    bar2 = plt.colorbar(c)
    bar2.set_label('$v$')
    plt.title(r'Vorticity with Streamlines $v$', fontsize = '8')
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    plt.streamplot(x, y, u[:, :, 0].transpose(), u[:,:,1].transpose(), color=[1,1,1], density = 0.87, linewidth = 0.4, arrowsize = 0.4)
    plt.savefig(f"{test_streamlines_dir}/Test_streamlines_{t}.png", dpi=300)
    plt.close()



    # TESTING SIM.MASK2 PLOT
    # To check if it is being initialised correctly
    colors = np.array([[1, 0, 0], [0, 0, 1]])  # Red for False, Blue for True
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    plt.imshow(sim.mask2.transpose(), cmap=cmap, origin='lower', extent=[0, sim.num_x, 0, sim.num_y]) # Plot the array
    cbar = plt.colorbar(ticks=[0, 1]) # Add colorbar for reference
    cbar.set_ticklabels(['False', 'True'])
    plt.title('sim.mask2 Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(f"{test_mask_dir}/Test_mask_{t}.png", dpi=300)
    plt.close()


    fig.clear()  # Clear the figure
    del fig, ax  # Delete references to objects
    plt.close('all')  # Close all figures
    gc.collect()  # Explicit garbage collection