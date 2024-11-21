#!/bin/bash
# ======================
# run_standard.sh
# ======================

#SBATCH --job-name=standard_job       # Name of the job
#SBATCH --partition=teach_cpu         # Use the teaching CPU partition
#SBATCH --account=PHYS033184          # Account for Advanced Computational Physics
#SBATCH --nodes=1                     # Use 1 node
#SBATCH --ntasks-per-node=1           # Use 1 task per node
#SBATCH --cpus-per-task=1             # Use 1 CPU per task
#SBATCH --time=23:30:00               # Wall time
#SBATCH --mem=5G                      # Memory allocation

NUM_RUNS=1  # Number of runs per thread count

# Remove leftover timings data
rm -rf *.txt

# Source the conda script to make conda command available
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the environment
conda activate LB_env

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

run_file=main.py

# Run the file NUM_RUNS times
for i in $(seq 1 $NUM_RUNS); do
    echo "Run $i"
    python $run_file
done