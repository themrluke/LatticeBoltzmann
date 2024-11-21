#!/bin/bash

# # Delete all .c and .so files in the current directory
# rm -rf build *.c *.so

# Source the conda script to make conda command available
source ~/miniconda3/etc/profile.d/conda.sh  # Adjust this path if needed for your Conda installation

# Activate the environment
conda activate LB_env

setup_file=setup.py
run_file=main.py

python $setup_file build_ext --inplace

# Specify the number of MPI processes (adjust as needed)
NUM_PROCESSES=8

# Run the Python program with MPI
mpiexec -n $NUM_PROCESSES python $run_file