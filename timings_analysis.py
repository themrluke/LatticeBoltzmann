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

def main():
    
    numba_times = find_min_times(filepath='numba/loop_timings.txt', num_runs=5, max_threads=28)
    openmp_times = find_min_times(filepath='openmp/loop_timings.txt', num_runs=5, max_threads=28)


