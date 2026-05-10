import subprocess
import time
import numpy as np
from pathlib import Path

def run_single_experiment(seed=42):
    """Run a single experiment and return the runtime."""
    start_time = time.perf_counter()
    
    # Run the main program with same parameters as run.sh
    result = subprocess.run([
        'python3', 'main.py',
        '--map', 'maps/1.txt',
        '--total_time', '600',
        '--num_iterations', '300', 
        '--algorithm', 'unique_vis',
        '--out', 'out/',
        '--seed', str(seed),
        '--no_plot',
        '--no_save'
    ], capture_output=True, text=True)
    
    end_time = time.perf_counter()
    runtime = end_time - start_time
    
    return runtime, result.returncode

def main():
    """Run benchmark and compute statistics."""
    num_runs = 50
    runtimes = []
    failed_runs = []
    
    print(f"Starting benchmark with {num_runs} runs...")
    print("Parameters: maps/1.txt, time=600, iterations=300, algorithm=unique_vis")
    
    # Create output directory
    Path("benchmark_results").mkdir(exist_ok=True)
    
    # Run experiments
    for i in range(1, num_runs + 1):
        print(f"Run {i}/{num_runs}...", end=" ")
        
        # Use different seeds for each run (same pattern as your experiments)
        runtime, return_code = run_single_experiment(seed=42 + i)
        
        if return_code == 0:
            runtimes.append(runtime)
            print(f"✓ {runtime:.3f}s")
        else:
            failed_runs.append(i)
            print(f"✗ Failed")
        
        # Progress indicator
        if i % 10 == 0:
            print(f"Completed {i} runs...")
    
    if failed_runs:
        print(f"Warning: {len(failed_runs)} runs failed: {failed_runs}")
    
    if not runtimes:
        print("Error: No successful runs to analyze!")
        return
    
    # Compute statistics
    runtimes = np.array(runtimes)
    mean_runtime = np.mean(runtimes)
    std_runtime = np.std(runtimes)
    min_runtime = np.min(runtimes)
    max_runtime = np.max(runtimes)
    median_runtime = np.median(runtimes)
    
    # Save raw data (compatible with your experiments format)
    with open('benchmark_results/runtime_data.txt', 'w') as f:
        for runtime in runtimes:
            f.write(f"{runtime:.6f}\n")
    
    # Save summary statistics
    with open('benchmark_results/runtime_statistics.txt', 'w') as f:
        f.write(f"Benchmark Results ({len(runtimes)} successful runs out of {num_runs})\n")
        f.write(f"Parameters: maps/1.txt, time=600, iterations=300, algorithm=unique_vis\n")
        f.write(f"\n")
        f.write(f"Mean Runtime: {mean_runtime:.4f} ± {std_runtime:.4f} seconds\n")
        f.write(f"Median Runtime: {median_runtime:.4f} seconds\n")
        f.write(f"Min Runtime: {min_runtime:.4f} seconds\n")
        f.write(f"Max Runtime: {max_runtime:.4f} seconds\n")
        if failed_runs:
            f.write(f"Failed runs: {len(failed_runs)} out of {num_runs}\n")
    
    # Print results (similar format to your experiments)
    print(f"\n=== Benchmark Results ({len(runtimes)} runs) ===")
    print(f"Mean Runtime: {mean_runtime:.4f} ± {std_runtime:.4f} seconds")
    print(f"Median Runtime: {median_runtime:.4f} seconds")
    print(f"Range: {min_runtime:.4f}s - {max_runtime:.4f}s")
    print(f"\nResults saved to: benchmark_results/")
    
    # Print in format compatible with your exec.ipynb analysis
    print(f"\nFor integration with experiments/exec.ipynb:")
    print(f"Mean: {mean_runtime}")
    print(f"Std: {std_runtime}")
    print(f"Count: {len(runtimes)}")

if __name__ == "__main__":
    main()