#!/bin/bash
# ======================
# run_mpi.sh
# ======================

#SBATCH --job-name=cython_job         # Name of the job
#SBATCH --partition=teach_cpu         # Use the teaching CPU partition
#SBATCH --account=PHYS033184          # Account for Advanced Computational Physics
#SBATCH --nodes=1                     # Use 1 node
#SBATCH --ntasks-per-node=1           # Use 1 task per node
#SBATCH --cpus-per-task=1             # Use 1 CPU per task
#SBATCH --time=0:02:00                # Wall time (10 minutes for testing)
#SBATCH --mem=5G                      # Memory allocation (1 GB)

# Parameters (set these variables)
MAX_PROCESSES=8         # Maximum number of MPI processes to test
NUM_RUNS_PER_PROCESS=2 # Number of runs per process count

# Remove leftover timings data
rm -rf *.txt

# Source the conda script to make conda command available
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the environment
conda activate LB_env

# # Change to the submission directory
# cd $SLURM_SUBMIT_DIR

setup_file=setup.py
run_file=main.py

python $setup_file build_ext --inplace

# Loop over process counts from 1 to MAX_PROCESSES
for processes in $(seq 1 $MAX_PROCESSES); do
    echo "Running with $processes MPI process(es)"
    
    # Run the file NUM_RUNS_PER_PROCESS times for each process count
    for i in $(seq 1 $NUM_RUNS_PER_PROCESS); do
        echo "Run $i for $processes MPI process(es)"
        mpiexec -n $processes python $run_file
    done
done