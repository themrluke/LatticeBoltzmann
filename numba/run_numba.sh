#!/bin/bash
# ======================
# run_numba.sh
# ======================

#SBATCH --job-name=numba_job         # Name of the job
#SBATCH --partition=teach_cpu         # Use the teaching CPU partition
#SBATCH --account=PHYS033184          # Account for Advanced Computational Physics
#SBATCH --nodes=1                     # Use 1 node
#SBATCH --ntasks-per-node=1           # Use 1 task per node
#SBATCH --cpus-per-task=28            # Use 28 CPU per task
#SBATCH --time=2:55:00                # Wall time
#SBATCH --mem=5G                      # Memory allocation

MAX_THREADS=28          # Maximum number of threads to test
NUM_RUNS_PER_THREAD=5   # Number of runs per thread count

# Remove leftover timings data
rm -rf *.txt

# Source the conda script to make conda command available
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the environment
conda activate LB_env

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

run_file=main.py

# Loop over thread counts from 1 to MAX_THREADS
for threads in $(seq 1 $MAX_THREADS); do
    # Set the number of threads for Numba
    export NUMBA_NUM_THREADS=$threads
    echo "Running with $threads thread(s)"

    # Run the file NUM_RUNS_PER_THREAD times for each thread count
    for i in $(seq 1 $NUM_RUNS_PER_THREAD); do
        echo "Run $i for $threads thread(s)"
        python $run_file
    done
done