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
THREADS=28               # Default number of threads if looping over num_x
NUM_X_VALUES=(2 4 6 8 10 20 40 60 80 120 250 400 600 800 1000 1200 1400 1600 2000 2400 2800 3200 4000 4800 5600 6400)  # Values of num_x to test
MODE="num_x"           # Options: "threads" or "num_x"


# Remove leftover timings data
rm -rf loop_timings.txt

# Source the conda script to make conda command available
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the environment
conda activate LB_env

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

run_file=main.py

if [ "$MODE" == "threads" ]; then
    # Loop over thread counts from 1 to MAX_THREADS
    for threads in $(seq 1 $MAX_THREADS); do
        # Set the number of threads for Numba
        export NUMBA_NUM_THREADS=$threads
        echo "Running with $threads thread(s)"

        # Run the file NUM_RUNS_PER_THREAD times for each thread count
        for i in $(seq 1 $NUM_RUNS_PER_THREAD); do
            echo "Run $i for $threads thread(s)"
            python $run_file --num_x 3200
        done
    done

elif [ "$MODE" == "num_x" ]; then
    # Loop over num_x values
    for num_x in "${NUM_X_VALUES[@]}"; do
        echo "Running with num_x=$num_x"

        # Set the number of threads for Numba
        export NUMBA_NUM_THREADS=$THREADS

        # Run the file NUM_RUNS_PER_THREAD times for each num_x value
        for i in $(seq 1 $NUM_RUNS_PER_THREAD); do
            echo "Run $i with num_x=$num_x and $THREADS Numba thread(s)"
            python $run_file --num_x $num_x
        done
    done

else
    echo "Invalid MODE specified. Use 'threads' or 'num_x'."
    exit 1
fi