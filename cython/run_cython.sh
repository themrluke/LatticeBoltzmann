#!/bin/bash
# ======================
# run_cython.sh
# ======================

#SBATCH --job-name=cython_job         # Name of the job
#SBATCH --partition=teach_cpu         # Use the teaching CPU partition
#SBATCH --account=PHYS033184          # Account for Advanced Computational Physics
#SBATCH --nodes=1                     # Use 1 node
#SBATCH --ntasks-per-node=1           # Use 1 task per node
#SBATCH --cpus-per-task=1             # Use 1 CPU per task
#SBATCH --time=2:00:00                # Wall time (20mins)
#SBATCH --mem=5G                      # Memory allocation

NUM_RUNS=5  # Number of runs per thread count
NUM_X_VALUES=(2 4 6 8 10 20 40 60 80 120 250 400 600 800 1000 1200 1400 1600 2000 2400 2800 3200 4000 4800 5600 6400)  # Values of num_x to test
MODE="num_x"           # Options: "one_size" or "num_x"

# Remove leftover timings data
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

if [ "$MODE" == "one_size" ]; then
    # Run the file NUM_RUNS times
    for i in $(seq 1 $NUM_RUNS); do
        echo "Run $i"
        python $run_file --num_x 3200
    done

elif [ "$MODE" == "num_x" ]; then
    # Loop over num_x values
    for num_x in "${NUM_X_VALUES[@]}"; do
        echo "Running with num_x=$num_x"

        # Run the file NUM_RUNS_PER_THREAD times for each num_x value
        for i in $(seq 1 $NUM_RUNS); do
            echo "Run $i with num_x=$num_x"
            python $run_file --num_x $num_x
        done
    done

else
    echo "Invalid MODE specified. Use 'threads' or 'num_x'."
    exit 1
fi