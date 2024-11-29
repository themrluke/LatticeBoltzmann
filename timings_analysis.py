# timings_analysis.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D


def find_min_times(filepath, num_runs, max_threads):
    """
    Finds the minimum run time for each repeat of each number of threads.

    Arguments:
        filepath (str): Path to the text file containing timing data
        num_runs (int): number of repeats for each number of threads
        max_threads (int): Maximum number of threads used

    Returns:
        min_times (list): list containing the fastest run for each number of threads
    """

    min_times = [] # Will hold the fastest run for each number of threads

    # Read data from text file
    with open(filepath, "r") as file:
        lines = file.readlines()

    # Check to ensure expected number of lines in file
    if len(lines) !=num_runs * max_threads:
        raise ValueError(f"Expected {num_runs * max_threads} lines, but found {len(lines)} in {filepath}")

    # Find the fastest run for each thread count
    for thread in range(max_threads):
        start_idx = thread * num_runs
        end_idx = start_idx + num_runs

        thread_times = [float(lines[i].strip()) for i in range(start_idx, end_idx)]

        min_times.append(min(thread_times)) # List now only includes fastest run for each thread count

    return min_times


def find_min_times_mpi(filepath, num_runs, max_threads):
    """
    Finds the minimum run time for each repeat for each number of threads for MPI.
    We need a different function here because MPI will print N timing values for N threads,
    we want only the slowest value (thread that finishes last) to be considered when calculating
    the fastest repeat at each number of threads.

    Arguments:
        filepath (str): Path to the text file containing timing data
        num_runs (int): number of repeats for each number of threads
        max_threads (int): Maximum number of threads used

    Returns:
        min_times (list): list containing the fastest run for each number of threads
    """

    min_times = []  # Holds the fastest run for each number of threads

    # Read data from text file
    with open(filepath, "r") as file:
        lines = file.readlines()

    current_index = 0

    # Find fastest run for each thread count
    for threads in range(1, max_threads + 1):

        run_times = []  # Stores the time for last process to finish for each thread count

        # Process each run for the current thread count
        for run in range(num_runs):
            # Extract the times for the current run
            run_timings = [
                float(lines[current_index + i].strip()) for i in range(threads)
            ]
            current_index += threads

            # We want the longest time (last process to finish)
            run_times.append(max(run_timings))

        # Find the minimum time across all runs for this thread count
        min_times.append(min(run_times))

    return min_times


def single_thread_times(filepath, num_runs):
    """
    For single threaded programs, we take the fastest value from the ones generated by the repeats.

    Arguments:
        filepath (str): Path to the text file containing timing data
        num_runs (int): number of repeats for each number of threads

    Returns:
        min_times (int): Fastest run time
    """

    min_time = []
    # Read the timing data from the text file
    with open(filepath, "r") as file:
        lines = file.readlines()

    # Quick check to ensure expected number of lines in file
    if len(lines) !=num_runs:
        raise ValueError(f"Expected {num_runs} lines, but found {len(lines)} in {filepath}")

    # Convert the lines to float and find the minimum value
    times = [float(line.strip()) for line in lines]
    min_time = min(times)

    return min_time


def find_avg_times(filepath, num_runs, max_threads):
    """
    Finds the average run time and standard deviation for each repeat of each number of threads.

    Arguments:
        filepath (str): Path to the text file containing timing data
        num_runs (int): number of repeats for each number of threads
        max_threads (int): Maximum number of threads used

    Returns:
        avg_times (list): List of average times for each thread count
        std_errors (list): List of standard deviations of the mean for each thread count
    """

    avg_times = []  # Holds the average run times
    std_errors = []  # Holds the standard deviations of the mean

    # Read data from text file
    with open(filepath, "r") as file:
        lines = file.readlines()

    # Check to ensure expected number of lines in file
    if len(lines) != num_runs * max_threads:
        raise ValueError(f"Expected {num_runs * max_threads} lines, but found {len(lines)} in {filepath}")

    # Calculate average and standard deviation for each thread count
    for thread in range(max_threads):
        start_idx = thread * num_runs
        end_idx = start_idx + num_runs

        thread_times = [float(lines[i].strip()) for i in range(start_idx, end_idx)]

        avg_times.append(np.mean(thread_times))  # Average time
        std_errors.append(np.std(thread_times) / np.sqrt(num_runs))  # Standard deviation of the mean

    return avg_times, std_errors


def find_avg_times_mpi(filepath, num_runs, max_threads):
    """
    Finds the average run time and standard deviation for MPI.

    Arguments:
        filepath (str): Path to the text file containing timing data
        num_runs (int): Number of repeats for each number of threads
        max_threads (int): Maximum number of threads used

    Returns:
        avg_times (list): List of average times for each thread count
        std_errors (list): List of standard deviations of the mean for each thread count
    """

    avg_times = []  # Holds the average run times
    std_errors = []  # Holds the standard deviations of the mean

    # Read data from text file
    with open(filepath, "r") as file:
        lines = file.readlines()

    expected_lines = sum(range(1, max_threads + 1)) * num_runs
    if len(lines) < expected_lines:
        raise ValueError(f"File contains {len(lines)} lines, but {expected_lines} are expected.")


    current_index = 0

    for threads in range(1, max_threads + 1):
        run_times = []  # Stores the time for the slowest process in each run

        for run in range(num_runs):
            run_timings = [
                float(lines[current_index + i].strip()) for i in range(threads)
            ]
            current_index += threads
            run_times.append(max(run_timings))  # Use the slowest process time

        avg_times.append(np.mean(run_times))
        std_errors.append(np.std(run_times) / np.sqrt(num_runs))

    return avg_times, std_errors


def find_avg_times_mpi_csv(filepath, num_runs, max_threads):
    """
    Finds the average run time and standard deviation for MPI from a CSV file.

    Arguments:
        filepath (str): Path to the CSV file containing timing data
        num_runs (int): Number of repeats for each number of threads
        max_threads (int): Maximum number of threads used

    Returns:
        avg_times (list): List of average times for each thread count
        std_errors (list): List of standard deviations of the mean for each thread count
    """

    # Initialize data structures
    data = {}  # Dictionary to store execution times grouped by threads and run_repeat

    # Read the CSV file
    with open(filepath, "r") as file:
        for line in file:
            # Split the line into columns
            cols = line.strip().split(",")
            execution_time = float(cols[0])
            rank = int(cols[1])
            num_x = int(cols[2])
            num_processes = int(cols[3])
            run_repeat = int(cols[4])

            # Group data by num_processes and run_repeat
            if num_processes not in data:
                data[num_processes] = {}
            if run_repeat not in data[num_processes]:
                data[num_processes][run_repeat] = []
            data[num_processes][run_repeat].append(execution_time)

    avg_times = []  # Holds the average run times
    std_errors = []  # Holds the standard deviations of the mean

    # Process data for each thread count
    for threads in range(1, max_threads + 1):
        if threads not in data:
            raise ValueError(f"No data found for {threads} threads.")

        run_times = []
        for run_repeat, timings in data[threads].items():
            if len(timings) != threads:
                raise ValueError(
                    f"Expected {threads} entries for run_repeat {run_repeat}, but found {len(timings)}."
                )
            run_times.append(max(timings))  # Take the slowest time for each repeat

        if len(run_times) != num_runs:
            raise ValueError(
                f"Expected {num_runs} run repeats for {threads} threads, but found {len(run_times)}."
            )

        avg_times.append(np.mean(run_times))  # Calculate mean for the thread count
        std_errors.append(np.std(run_times) / np.sqrt(num_runs))  # Standard error

    return avg_times, std_errors



def find_avg_times_system_sizes(filepath, num_runs, system_sizes):
    """
    Finds the average run time and standard deviation for each system size.

    Arguments:
        filepath (str): Path to the text file containing timing data
        num_runs (int): Number of repeats for each system size
        system_sizes (list): List of system sizes

    Returns:
        avg_times (list): List of average times for each system size
        std_errors (list): List of standard deviations of the mean for each system size
    """

    avg_times = []  # Holds the average run times for each system size
    std_errors = []  # Holds the standard deviation of the mean for each system size

    # Read data from text file
    with open(filepath, "r") as file:
        lines = file.readlines()

    # Check to ensure expected number of lines in file
    if len(lines) != num_runs * len(system_sizes):
        raise ValueError(f"Expected {num_runs * len(system_sizes)} lines, but found {len(lines)} in {filepath}")

    # Calculate average and standard deviation for each system size
    for i, size in enumerate(system_sizes):
        start_idx = i * num_runs
        end_idx = start_idx + num_runs

        size_times = [float(lines[j].strip()) for j in range(start_idx, end_idx)]

        avg_times.append(np.mean(size_times))  # Average time
        std_errors.append(np.std(size_times) / np.sqrt(num_runs))  # Standard deviation of the mean

    return avg_times, std_errors


def find_avg_times_system_sizes_mpi(filepath, num_runs, system_sizes, num_processes):
    """
    Finds the average run time and standard deviation for each system size for MPI runs.

    For MPI runs, where there are `num_processes` outputs per run, we consider only the slowest
    time (representing the last process to finish) for the average calculation.

    Arguments:
        filepath (str): Path to the text file containing timing data
        num_runs (int): Number of repeats for each system size
        system_sizes (list): List of system sizes
        num_processes (int): Number of processes used in the MPI runs

    Returns:
        avg_times (list): List of average times for each system size
        std_errors (list): List of standard deviations of the mean for each system size
    """

    avg_times = []  # Holds the average run times for each system size
    std_errors = []  # Holds the standard deviation of the mean for each system size

    # Read and parse the CSV data
    data = []
    with open(filepath, "r") as file:
        for line in file:
            cols = line.strip().split(",")
            execution_time = float(cols[0])
            rank = int(cols[1])
            num_x = int(cols[2])
            num_processes = int(cols[3])
            run_repeat = int(cols[4])
            data.append((execution_time, rank, num_x, num_processes, run_repeat))

    # Filter the data to ensure it matches the given number of processes
    data = [row for row in data if row[3] == num_processes]

    # Validate the number of runs and system sizes
    for size in system_sizes:
        size_data = [row for row in data if row[2] == size]
        if len(size_data) != num_runs * num_processes:
            raise ValueError(f"Expected {num_runs * num_processes} entries for system size {size}, but found {len(size_data)}.")

    # Process data for each system size
    for size in system_sizes:
        size_times = []  # Store the slowest time for each run of the current system size

        for run in range(1, num_runs + 1):
            # Extract times for the current system size and run repeat
            run_data = [row[0] for row in data if row[2] == size and row[4] == run]
            if len(run_data) != num_processes:
                raise ValueError(f"Expected {num_processes} entries for system size {size} and run {run}, but found {len(run_data)}.")

            # Append the slowest time (last process to finish)
            size_times.append(max(run_data))

        # Calculate average and standard deviation of the slowest times
        avg_times.append(np.mean(size_times))
        std_errors.append(np.std(size_times) / np.sqrt(num_runs))

    return avg_times, std_errors


def speedup_plot(implementations, max_threads, name, overlays=None):
    """
    Dynamically plots speedup and efficiency for multiple implementations

    Arguments:
        implementations (list of tuples): Each tuple contains:
            - avg_times (list): Average execution times
            - std_errors (list): Standard errors for the execution times
            - label (str): Label for the implementation
            - color (str): Color for the plot
        max_threads (int): Maximum number of threads used
        name (str): Filename to save plot as
        overlays (list of tuples): Additional functions to plot for comparison
    """
    threads = np.arange(1, max_threads + 1)

    fig, ax1 = plt.subplots(figsize=(10, 8))

    # Lists to store efficiency data for the secondary y-axis
    efficiency_lines = []

    # Loop through implementations
    for avg_times, std_errors, label, color in implementations:
        # Calculate speedup
        speedup = [avg_times[0] / t for t in avg_times]
        speedup_err = [std_errors[0] / t for t in avg_times]  # Propagate error (approximation)

        # Calculate efficiency
        efficiency = [s / w for s, w in zip(speedup, threads)]
        efficiency_err = [se / w for se, w in zip(speedup_err, threads)]

        # Plot speedup
        ax1.errorbar(
            threads, speedup, yerr=speedup_err, label=f"{label}",
            color=color, linewidth=1.5, marker="o", markersize=3, capsize=3
        )

        # Store efficiency data for the secondary y-axis
        efficiency_lines.append((efficiency, efficiency_err, color))

    # Overlay raw data (e.g., remainder line)
    if overlays:
        for x_values, y_values, label, color in overlays:
            ax1.plot(x_values, y_values, label=label, color=color, linewidth=1.5)


    # Configure the speedup axis
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(2)  # Thicken left spine
    ax1.spines['bottom'].set_linewidth(2)  # Thicken bottom spine
    ax1.set_xlabel("Workers", fontsize=25)
    ax1.set_ylabel("Speedup", fontsize=25)
    ax1.set_xscale('linear')
    ax1.set_yscale('linear')
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.tick_params(axis='both', which='major', width=2, length=8,  labelsize=20)

    # Create secondary axis for efficiency
    ax2 = ax1.twinx()

    for efficiency, efficiency_err, color in efficiency_lines:
        ax2.errorbar(
            threads, efficiency, yerr=efficiency_err, linestyle='dashed',
            color=color, linewidth=2, alpha=0.4, marker="o", markersize=2, capsize=3
        )

    # Configure the efficiency axis
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.set_ylabel("Efficiency", fontsize=25)
    ax2.set_yscale('linear')
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(ScalarFormatter())
    ax2.tick_params(axis='both', which='major', width=2, length=8,  labelsize=20)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # Create custom legend entries for line styles
    custom_lines = [
        Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Speedup'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Efficiency')
    ]

    # Combine all legend entries
    ax1.legend(
        custom_lines + lines1 + lines2,
        [line.get_label() for line in custom_lines] + labels1 + labels2,
        loc='center right',
        #loc='lower right',
        fontsize=20
    )

    # Add grid and layout adjustments
    #ax1.grid(True, which="both", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{name}.png", dpi=300)
    plt.close()


def times_plot(implementations, max_threads, name, overlays=None):
    """
    Dynamically plots raw execution times for multiple implementations.

    Arguments:
        implementations (list of tuples): Each tuple contains:
            - avg_times (list): Average execution times
            - std_errors (list): Standard errors for the execution times
            - label (str): Label for the implementation
            - color (str): Color for the plot
        max_threads (int): Maximum number of threads used
        name (str): Filename to save plot as
        overlays (list of tuples): Additional functions to plot for comparison
    """

    threads = np.arange(1, max_threads + 1)

    fig, ax1 = plt.subplots(figsize=(10, 8))

    # Plot raw times for each implementation
    for avg_times, std_errors, label, color in implementations:
        ax1.errorbar(
            threads, avg_times, yerr=std_errors, label=label,
            color=color, linewidth=1.5, marker="o", markersize=3, capsize=3,
        )

    # Overlay raw data (e.g., remainder line)
    if overlays:
        for x_values, y_values, label, color in overlays:
            ax1.plot(x_values, y_values, label=label, color=color, linewidth=1.5)

    # Configure the axes
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(2)  # Thicken left spine
    ax1.spines['bottom'].set_linewidth(2)  # Thicken bottom spine
    ax1.tick_params(axis='both', which='major', width=2, length=8,  labelsize=20)  # Adjust tick length too
    ax1.tick_params(axis='both', which='minor', width=1.5, length=4,  labelsize=20)
    ax1.set_xlabel("Workers", fontsize=25)
    ax1.set_ylabel("Time (s)", fontsize=25)
    ax1.set_xscale('linear')
    ax1.set_yscale('log')

    # Combine legends for raw times and overlays
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper right', fontsize=20)

    # Add layout adjustments
    plt.tight_layout()
    plt.savefig(f"{name}.png", dpi=300)
    plt.close()



def system_size_plot(system_sizes, implementations):
    """
    Line plot to show execution time vs. system size for multiple implementations.

    Arguments:
        system_sizes (list): List of system sizes (e.g., num_x values).
        implementations (list of tuples): Each tuple contains:
            - avg_times (list): Average execution times.
            - std_errors (list): Standard errors for the execution times.
            - implementation_name (str): Name of the implementation.
    """

    plt.figure(figsize=(10, 8))

    for avg_times, std_errors, implementation_name, color in implementations:
        plt.errorbar(
            system_sizes, avg_times, yerr=std_errors, label=implementation_name,
            linewidth=1.5, color=color, marker="o", markersize=3, capsize=3
        )

    plt.xlabel("System Size (num_x)", fontsize=25)
    plt.ylabel("Time (s)", fontsize=25)
    plt.yscale('log')
    plt.xscale('log')

    # Set integer formatting for both axes
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)  # Thicken left spine
    ax.spines['bottom'].set_linewidth(2)  # Thicken bottom spine
    ax.tick_params(axis='both', which='major', width=2, length=8,  labelsize=20)  # Adjust tick length too
    ax.tick_params(axis='both', which='minor', width=1.5, length=4,  labelsize=20)
    ax.ylim = (0.1, 1000)
    ax.xlim = (1, 1000)

    plt.legend(fontsize=20)
    plt.grid(axis='y', zorder=0, linewidth=0.5)
    plt.tight_layout()
    plt.savefig("system_sizes.png", dpi=300)
    plt.close()


def bar_plot(standard_time, vectorised_time, cython_time, mpi_times, openmp_times, numba_times, numba_gpu_time, numba_intel_min_time):
    """
    Creates a bar plot comparing the execution times of standard, vectorised, and Cython implementations.

    Args:
        standard_time (float): Execution time for standard Python program
        vectorised_time (float): Execution time for vectorised Python program
        cython_time (float): Execution time for Cython program
    """

    openmp_1thread = openmp_times[0]
    openmp_8threads = openmp_times[7]
    openmp_28threads = openmp_times[27]

    numba_1thread = numba_times[0]
    numba_8threads = numba_times[7]
    numba_28threads = numba_times[27]

    mpi_1process = mpi_times[0]
    mpi_8processes = mpi_times[7]
    mpi_28processes = mpi_times[27]

    # Labels and times
    labels = [
        'Standard Python', 'vectorised Python', 'Cython',
        'OpenMP (1 Thread)', 'Numba (1 Thread)', ' MPI (1 Process)',
        'OpenMP (8 Threads)', 'Numba (8 Threads)', ' Numba (8 Threads)', ' MPI (8 Processes)',
        'OpenMP (28 Threads)', 'Numba (28 Threads)', ' MPI (28 Processes)',
        'Numba GPU (4070TI)'
    ]
    times = [
        standard_time, vectorised_time, cython_time,
        openmp_1thread, numba_1thread, mpi_1process,
        openmp_8threads, numba_8threads, numba_intel_min_time, mpi_8processes,
        openmp_28threads, numba_28threads, mpi_28processes,
        numba_gpu_time
    ]

    # Create the bar plot
    plt.figure(figsize=(10, 8))
    bars = plt.bar(labels, times, width=0.75, align='center', zorder=3)

    # Set all bars to blue initially
    for bar in bars:
        bar.set_color('#1f77b4')

    # Set the last bar to orange
    bars[8].set_color('purple')
    bars[-1].set_color('orange')

    # Add a legend for the bar colors
    plt.legend(
        ['BC4 (BlueCrystal 4)', 'GPU (4070TI)'],
        loc='upper right', fontsize=20, frameon=True,
        handles=[
            plt.Line2D([0], [0], color='#1f77b4', lw=6, label='CPU (BlueCrystal 4)'),
            plt.Line2D([0], [0], color='purple', lw=6, label='CPU (Intel i7 13700K)'),
            plt.Line2D([0], [0], color='orange', lw=6, label='GPU (4070Ti)')
        ]
    )

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)  # Thicken left spine
    ax.spines['bottom'].set_linewidth(2)  # Thicken bottom spine
    ax.tick_params(axis='both', which='major', width=2, length=8,  labelsize=20)  # Adjust tick length too
    ax.tick_params(axis='y', which='minor', width=1.5, length=4,  labelsize=20)

    plt.grid(axis='y', zorder=0, linewidth=0.5)
    plt.yscale('log')  # Logarithmic scale for the y-axis
    plt.ylim(0.5, 2.4 * 10**4)  # Adjust the starting and ending values of the y-axis
    plt.ylabel("Time (s)", fontsize=25)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=15)  # Rotate 45 degrees and align to the right

    # Add the values inside the bars
    for bar, time in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width() / 2.0, 0.62,  # Position inside the bar
                 f"{time:.2f}", ha='center', va='baseline', color='white', fontsize=22, fontweight='bold', rotation='vertical')

    plt.tight_layout()

    # Save the plot
    plt.savefig("bar_plot.png", dpi=300)
    plt.close()


def main():

    # Average times across all runs for each thread count
    numba_avg, numba_err = find_avg_times(filepath='numba/loop_timings_speedup.txt', num_runs=5, max_threads=28)
    openmp_avg_static, openmp_err_static = find_avg_times(filepath='openmp/loop_timings_speedup_static.txt', num_runs=5, max_threads=28)
    openmp_avg_dynamic, openmp_err_dynamic = find_avg_times(filepath='openmp/loop_timings_speedup_dynamic.txt', num_runs=5, max_threads=28)
    openmp_avg_guided, openmp_err_guided = find_avg_times(filepath='openmp/loop_timings_speedup_guided.txt', num_runs=5, max_threads=28)
    mpi_avg, mpi_err = find_avg_times_mpi(filepath='mpi/loop_timings_speedup.txt', num_runs=5, max_threads=28)
    mpi_avg_2nodes, mpi_err_2nodes = find_avg_times_mpi_csv(filepath='mpi/loop_timings_speedup_2nodes.csv', num_runs=5, max_threads=56)

    # Implementations for speedup plotting
    implementations_speedup = [
        (numba_avg, numba_err, "Numba", 'red'),
        (openmp_avg_guided, openmp_err_guided, "OpenMP", 'blue'),
        (mpi_avg, mpi_err, "MPI", 'limegreen')
    ]

    speedup_plot(implementations_speedup, max_threads=28, name='speedup')
    times_plot(implementations_speedup, max_threads=28, name='times')

    # Make a plot of the MPI speedup across 2 nodes
    mpi_speedup_2nodes = [
        (mpi_avg_2nodes, mpi_err_2nodes, "MPI (2 Nodes)", 'crimson')
    ]

    # Overlay the remainder contribution for each amount of workers
    remainder_line = -0.01 * np.arange(1, 57, 1) * np.array([3200 % N for N in range(1, 57)]) + np.arange(1, 57, 1)
    overlays = [
        (np.arange(1, 57), remainder_line, "Load Imbalance", 'grey')
    ]

    speedup_plot(mpi_speedup_2nodes, max_threads=56, name='mpi_2_nodes_speedup', overlays=overlays)

    # Compare the speedups of different OpenMP schedules
    implementations_openmp = [
        (openmp_avg_static, openmp_err_static, "OpenMP static", 'magenta'),
        (openmp_avg_dynamic, openmp_err_dynamic, "OpenMP dynamic", 'purple'),
        (openmp_avg_guided, openmp_err_guided, "OpenMP guided", 'blue'),
    ]

    speedup_plot(implementations_openmp, max_threads=28, name='openmp_schedules')


    # Minimum times for bar plot
    numba_min_threads = find_min_times(filepath='numba/loop_timings_speedup.txt', num_runs=5, max_threads=28)
    openmp_min_threads = find_min_times(filepath='openmp/loop_timings_speedup_guided.txt', num_runs=5, max_threads=28)
    mpi_min_threads = find_min_times_mpi(filepath='mpi/loop_timings_speedup.txt', num_runs=5, max_threads=28)
    standard_min_time = single_thread_times(filepath='standard_python/loop_timings_3200.txt', num_runs=1)
    vectorised_min_time = single_thread_times(filepath='vectorised_python/loop_timings_3200.txt', num_runs=10)
    cython_min_time = single_thread_times(filepath='cython/loop_timings_3200.txt', num_runs=10)
    numbagpu_min_time = single_thread_times(filepath='numba_gpu/loop_timings_3200.txt', num_runs=5)

    # Minimum time across all runs on Intel
    numba_intel_min_time = single_thread_times(filepath='numba/loop_timings_intel.txt', num_runs=5)

    bar_plot(
        standard_min_time, vectorised_min_time, cython_min_time,
        mpi_min_threads, openmp_min_threads, numba_min_threads,
        numbagpu_min_time, numba_intel_min_time
    )


    # Below is plot for changing system sizes
    system_sizes = [ # x-sizes of lattice
        2, 4, 6, 8, 10, 20, 40, 60, 80, 120, 250, 400, 600, 800, 1000, 1200, 1400, 1600, 2000, 2400, 2800, 3200, 4000, 4800, 5600, 6400
    ]

    num_runs = 5  # Number of repeats for each system size

    # Parse timing data for system sizes
    numba_avg_sizes, numba_err_sizes = find_avg_times_system_sizes(
        filepath='numba/loop_timings_sizes.txt', num_runs=num_runs, system_sizes=system_sizes
    )
    openmp_avg_sizes, openmp_err_sizes = find_avg_times_system_sizes(
        filepath='openmp/loop_timings_sizes.txt', num_runs=num_runs, system_sizes=system_sizes
    )
    mpi_avg_sizes, mpi_err_sizes = find_avg_times_system_sizes_mpi(
        filepath='mpi/loop_timings_sizes.csv', num_runs=num_runs, system_sizes=system_sizes, num_processes=28
    )
    numbagpu_avg_sizes, numbagpu_err_sizes = find_avg_times_system_sizes(
        filepath='numba_gpu/loop_timings_sizes.txt', num_runs=num_runs, system_sizes=system_sizes
    )
    cython_avg_sizes, cython_err_sizes = find_avg_times_system_sizes(
        filepath='cython/loop_timings_sizes.txt', num_runs=num_runs, system_sizes=system_sizes
    )
    vectorized_avg_sizes, vectorized_err_sizes = find_avg_times_system_sizes(
        filepath='vectorised_python/loop_timings_sizes.txt', num_runs=num_runs, system_sizes=system_sizes
    )

    implementations_sizes = [
        (numba_avg_sizes, numba_err_sizes, "Numba", 'red'),
        (openmp_avg_sizes, openmp_err_sizes, "OpenMP", 'blue'),
        (mpi_avg_sizes, mpi_err_sizes, "MPI", 'limegreen'),
        (numbagpu_avg_sizes, numbagpu_err_sizes, "GPU (4070TI)", 'orange'),
        (cython_avg_sizes, cython_err_sizes, "Cython", 'purple'),
        (vectorized_avg_sizes, vectorized_err_sizes, "Vectorized Python", 'turquoise'),
    ]

    # Generate the combined plot
    system_size_plot(system_sizes, implementations_sizes)


if __name__ == "__main__":
    main()