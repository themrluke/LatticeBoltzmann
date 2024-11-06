#!/bin/bash

# Delete all .c files in the current directory
rm -f *.c

# Source the conda script to make conda command available
source ~/miniconda3/etc/profile.d/conda.sh  # Adjust this path if needed for your Conda installation

# Activate the environment
conda activate LB_env

setup_file=setup.py
run_file=main.py

python $setup_file build_ext --inplace
python $run_file