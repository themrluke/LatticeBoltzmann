import numpy as np
import matplotlib.pyplot as plt

def find_min_times(filepath, num_runs, max_threads):
    
    min_times = [] # Will hold the fastest run for each number of threads

    # Read the timing data from the text file
    with open(filepath, "r") as file:
        lines = file.readlines()

    # Quick check to ensure expected number of lines in file
    if len(lines) !=num_runs * max_threads:
        raise ValueError(f"Expected {num_runs * max_threads} lines, but found {len(lines)} in {filepath}")
    
    # Find the fastes run for each thread count
    for thread in range(max_threads):
        start_idx = thread * num_runs
        end_idx = start_idx + num_runs

        thread_times = [float(lines[i].strip()) for i in range(start_idx, end_idx)]

        min_times.append(min(thread_times)) # List now only inclues fastest run for each thread count

    return min_times

def find_min_times_mpi(filepath, num_runs, max_threads):
    """
    Finds the minimum run time for each number of threads for MPI.
    
    Args:
        filepath (str): Path to the text file containing timings.
        num_runs (int): Number of runs per thread count.
        max_threads (int): Maximum number of threads tested.
        
    Returns:
        list: Minimum run time for each thread count.
    """
    min_times = []  # Holds the fastest run for each number of threads

    # Read the timing data from the text file
    with open(filepath, "r") as file:
        lines = file.readlines()

    current_index = 0

    # Process each thread count from 1 to max_threads
    for threads in range(1, max_threads + 1):
        run_times = []  # Stores the longest time for each run

        # Process each run for the current thread count
        for run in range(num_runs):
            # Extract the times for the current run
            run_timings = [
                float(lines[current_index + i].strip()) for i in range(threads)
            ]
            current_index += threads

            # Select the longest time from this run
            run_times.append(max(run_timings))

        # Find the minimum time across all runs for this thread count
        min_times.append(min(run_times))

    return min_times


def main():
    
    numba_times = find_min_times(filepath='numba/loop_timings.txt', num_runs=5, max_threads=28)
    openmp_times = find_min_times(filepath='openmp/loop_timings.txt', num_runs=5, max_threads=28)
    mpi_times = find_min_times_mpi(filepath='mpi/loop_timings.txt', num_runs=5, max_threads=28)

    threads = np.arange(1, 29)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(threads, numba_times, label="Numba", marker="o")
    plt.plot(threads, openmp_times, label="OpenMP", marker="s")
    plt.plot(threads, mpi_times, label="MPI", marker="^")
    plt.xlabel("Number of Threads")
    plt.ylabel("Minimum Execution Time (seconds)")
    plt.title("Minimum Execution Time vs Number of Threads (Logarithmic Scale)")
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Grid for both major and minor ticks
    plt.tight_layout()
    plt.savefig("timings_plots_log.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()