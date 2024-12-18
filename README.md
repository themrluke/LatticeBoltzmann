<div align="center">
  <h1><strong>Advanced Computational Physics <br> Miniproject</strong></h1>
  
  <p><strong>Written by: Luke Johnson (sa21722)</strong></p>
  <p><strong>Updated: 18/12/2024</strong></p>
  <p><strong>Repository for the Advanced Computational Physics miniproject! :)</strong></p>
</div>

<br>
<br>

# Getting started

This **README** is designed to be rendered by GitHub or VScode with `ctrl + shift + v`.

The repo contains code for 2D fluid dynamics simulation using the Lattice Boltzmann Method (LBM).

Although there are many choices of obstacle, the investigation focussed on measuring drag forces for a car in a slipstream. 

# Setting up the env

Many different libraries were used for parallelization and compiling Python code. Follow this setup on a Linux node, Bristol **bc4login.acrc.bris.ac.uk** is suitable. The GPU code was run on a Windows machine using [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) using an [NVIDIA RTX 4070 Ti](https://www.nvidia.com/en-gb/geforce/graphics-cards/40-series/rtx-4070-family/)

1. Clone this repository
2. The environment [file](env.yaml) contains all necessary libraries
3. Create the Conda environment with `conda env create -f env.yaml`
4. The preferred method of interaction with this repository is via [VSCode](https://code.visualstudio.com/download). Download any extensions required as prompted.

<br>

# Repo Layout

**[slipstream_force_DATA](slipstream_force_DATA/)**
- This folder contains the transverse force data for a car in a slipstream at various following distances
- These `.csv` files were imported into OriginPro for data analysis

<br>

**[GRAPHS](GRAPHS/)**
- Force evolution through time for different slipstream distances
- Average force at each car separation distance
- Execution time for different implementations
- Execution time for different number of threads/processes
- Speedup and efficiency of different implementations and OpenMP schedules
- Execution time for a variety of system sizes
- Vortices plots were not included in repo but can be seen by:
    1. Creating new ones by running the simulation and adjusting the `t_plot` interval
    2. Looking at an example plot [here](GRAPHS/streamlines128_24.png)

<br>

**[masks](masks/)**
- Binary `.txt` files containing obstacle data
- 1 = obstacle, 0 = empty lattice site
- This method allows visual feedback for users while drawing complex mask shapes
- Mask data is read into Python scripts and overalyed on a custom region of lattice

<br>

&#9642; The [timings_analysis](timings_analysis.py) file is responsible for creating the speedup and execution time [plots](GRAPHS/)

&#9642; This [pdf](Advanced_Computational_Physics_Report.pdf) has a formally written account of the results and findings of the investigation

&#9642; There is also a rendered [video](vortices_video.mp4) showing the simulation in action

&#9733; The other folders, containing the simulation code for each implementation, are explained in the **Recipe** below

<br>

# Large files

- It's a good idea to store large video files with git Large File System (git LFS).
    - you can download git LFS with:
    ```bash
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh 
    mkdir -p ~/bin
    export PATH=~/bin:$PATH
    apt-get download git-lfs
    dpkg-deb -x git-lfs_* ~/bin
    ```
    - and add it your $PATH with:
    ```bash
    export PATH="$HOME/bin/usr/local/git-lfs/bin:$PATH"
    ```

- init git lfs in the repo with:
```git lfs install```

Then you can use the standard git commands.

<br>
<br>
<br>

<div align="center">
  <h1><strong>RECIPE</strong></h1>
</div>

&#9642; Each implementation has its own self-contained folder. The only exception is the [mask](masks/) data which is shared between them.

&#9642; In each folder there are various abstractions defined in standalone python modules

&#9642; `.txt` files in each subfolder contain the results for execution times of various repeats

&#9642; Additions should be pushed to a new branch followed by a Merge Request to master

The different implementations are:
1. [Standard Python](standard_python/)
2. [Vectorized Python](vectorized_python/)
3. [Cython](cython/)
4. [OpenMP](openmp/)
5. [Numba](numba/)
6. [MPI](mpi/)
7. [GPU](numba_gpu/)

**&#9733; To run the code, navigate to one of these folders and type: `./run_<implementation>.sh`,
<br>
making the `.sh` file is configured to the correct threads/repeats/lattice size etc.
<br>
<br>
&#9733; If using a job scheduler like Slurm, then use: `sbatch run_<implementation>.sh` instead**

&#9642; Each of these folders will contain the following files:


## 1. Shell Script

- Use this file to run the simulation

- Named `run_<implementation>.sh`

- This file can be submitted to **Slurm** queuing systems

- The number of repeats/threads/processes can all be adjusted here, along with the system size

- Also responsible for building any setup files


## 2. Setup File

- Only the [Cython](cython/), [OpenMP](openmp/), [MPI](mpi/) versions will have this file

- Specifies how to build, compile, and install Cython modules with optimisations

- Converts `.pyx` files into compiled C extensions (`.so` files)

- Configures build settings using MPI and OpenMP

- Enables high-performance compiler flags

- Manages dependencies to ensure compatibility



## 3. Main File

- Imports pre-coded Python modules

- Sets up simulation parameters

- Runs the initialisation step

- Evolves the fluid dynamics simulation using a time-stepping loop

- Saves plots in specified intervals

- Gathers simulation force data and saves to `.csv` file

**NOTE:** In the [MPI version](mpi/main.py), it also sets up the ranks and divides workload. The [GPU version](numba_gpu/main.py) configures the threads per block, blocks per grid, and  launches the CUDA kernels.


## 4. Parameters File

- Contains a class which sets key fluid simulation parameters like grid size

- Defines the D2Q9 lattice model velocity directions `c`, weight coefficients `w`, and reflection mapping

**NOTE:** The [Cython](cython/), [OpenMP](openmp/), [MPI](mpi/) versions used a Cythonized version of this file which requires a separate declaration `.pxd` file. This defines the structure and attributes of the C class which allows the Cython compiler to generate efficient C extensions.


## 5. Initialisation File

- Sets up obstacles inline with the user's choice

- Reads the binary obstacle masks from corresponding filepath and overlays them onto the grid

- Handles special obstacle setups, such as adding the road and tunnel for the car simulations

- Initialises the fluid density and velocity fields


## 6. Fluid Dynamics File

- Contains the core Lattice Boltzmann equations and logic for the simulation

- The Cythonized versions of this file will include the timestep loop to evolve fluid properties over time

- Includes functions for: collision, streaming & boundary reflection, fluid density, velocity, vorticity, and equilibrium distribution