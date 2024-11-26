#!/bin/bash
# ======================
# run_mpi.sh
# ======================

#SBATCH --job-name=mpi_job            # Name of the job
#SBATCH --partition=cpu
#SBATCH --account=PHYS033184          # Account for Advanced Computational Physics
#SBATCH --nodes=1                     # Use 1 node
#SBATCH --ntasks-per-node=28          # Use 28 task per node
#SBATCH --cpus-per-task=1             # Use 1 CPU per task
#SBATCH --time=10:30:00               # Wall time
#SBATCH --mem=10G                      # Memory allocation

# Parameters (set these variables)
MAX_PROCESSES=28         # Maximum number of MPI processes to test
NUM_RUNS_PER_PROCESS=5   # Number of runs per process count
PROCESSES=28               # Default number of threads if looping over num_x
NUM_X_VALUES=(120 250 400 600 800 1000 1200 1400 1600 2000 2400 2800 3200 4000 4800 5600 6400)  # Values of num_x to test
MODE="threads"           # Options: "threads" or "num_x"

# Remove leftover timings data
rm -rf *.out
rm -rf loop_timings.txt

# Source the conda script to make conda command available
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the environment
conda activate LB_env

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

setup_file=setup.py
run_file=main.py

python $setup_file build_ext --inplace

if [ "$MODE" == "threads" ]; then
    # Loop over process counts from 1 to MAX_PROCESSES
    for processes in $(seq 1 $MAX_PROCESSES); do
        echo "Running with $processes MPI process(es)"

        # Run the file NUM_RUNS_PER_PROCESS times for each process count
        for i in $(seq 1 $NUM_RUNS_PER_PROCESS); do
            echo "Run $i for $processes MPI process(es)"
            mpiexec -n $processes python $run_file --num_x 3200
        done
    done

elif [ "$MODE" == "num_x" ]; then
    # Loop over num_x values
    for num_x in "${NUM_X_VALUES[@]}"; do
        echo "Running with num_x=$num_x"

        # Run the file NUM_RUNS_PER_THREAD times for each num_x value
        for i in $(seq 1 $NUM_RUNS_PER_PROCESS); do
            echo "Run $i with num_x=$num_x and $PROCESSES MPI process(es)"
            mpiexec -n $PROCESSES python $run_file --num_x $num_x
        done
    done

else
    echo "Invalid MODE specified. Use 'threads' or 'num_x'."
    exit 1
fi