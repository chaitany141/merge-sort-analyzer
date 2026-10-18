"""
benchmark.py
============
Benchmarking utilities, distribution data generators, and comparative performance
metrics calculation for Merge Sort and Multithreaded/Multiprocess Merge Sort.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from algorithms import merge_sort, multithreaded_merge_sort


DISTRIBUTIONS = [
    "Random",
    "Sorted (Best Case)",
    "Reverse Sorted (Worst Case)",
    "Nearly Sorted",
]


def generate_data(
    size: int, distribution: str = "Random", seed: Optional[int] = None
) -> List[int]:
    """
    Generates synthetic datasets with controlled distribution properties.

    Distributions:
        - "Random": Uniformly distributed random integers.
        - "Sorted (Best Case)": Strictly ascending sequence.
        - "Reverse Sorted (Worst Case)": Strictly descending sequence.
        - "Nearly Sorted": Ascending sequence with ~5% random perturbations.

    Parameters:
        size: Number of elements.
        distribution: Distribution category name.
        seed: Optional RNG seed for reproducibility.

    Returns:
        List of integers.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    dist_normalized = distribution.strip().lower()

    if "sorted" in dist_normalized and "reverse" not in dist_normalized and "nearly" not in dist_normalized:
        return list(range(size))

    if "reverse" in dist_normalized:
        return list(range(size, 0, -1))

    if "nearly" in dist_normalized:
        arr = list(range(size))
        # Perform swaps on ~5% of elements (at least 1, at most 250)
        num_swaps = max(1, min(int(size * 0.05), 250))
        for _ in range(num_swaps):
            i = random.randint(0, size - 1)
            j = random.randint(0, size - 1)
            arr[i], arr[j] = arr[j], arr[i]
        return arr

    # Default: Random
    # Using numpy for high-speed generation of large datasets
    return np.random.randint(-size * 5, size * 5 + 1, size=size).tolist()


def time_algorithm(
    sort_func: Callable[..., List[Any]],
    arr: List[Any],
    trials: int = 3,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Executes multiple trials of a sorting algorithm on identical copies of data.

    Parameters:
        sort_func: The sorting function to benchmark.
        arr: The base list to be sorted.
        trials: Number of benchmark repetitions.
        **kwargs: Extra arguments passed to sort_func (e.g. num_threads, executor_type).

    Returns:
        Dictionary containing mean, min, max, std dev, and individual trial times.
    """
    times: List[float] = []
    last_result: Optional[List[Any]] = None

    for _ in range(max(1, trials)):
        # Provide a clean, independent copy to prevent in-place modification
        data_copy = list(arr)
        t_start = time.perf_counter()
        last_result = sort_func(data_copy, **kwargs)
        t_end = time.perf_counter()
        times.append(t_end - t_start)

    mean_t = float(np.mean(times))
    min_t = float(np.min(times))
    max_t = float(np.max(times))
    std_t = float(np.std(times)) if len(times) > 1 else 0.0

    return {
        "mean_time": mean_t,
        "min_time": min_t,
        "max_time": max_t,
        "std_time": std_t,
        "trials": times,
        "result": last_result,
    }


def run_comparison(
    size: int,
    distribution: str,
    num_threads: int,
    trials: int = 3,
    executor_type: str = "thread",
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Performs a head-to-head performance comparison between single-threaded
    and multithreaded/multiprocess merge sort on the same input dataset.

    Returns:
        Dictionary with benchmark results, speedup, verification assert flags, and summary data.
    """
    data = generate_data(size=size, distribution=distribution, seed=seed)

    # Benchmark Single-Threaded Merge Sort
    seq_metrics = time_algorithm(merge_sort, data, trials=trials)

    # Benchmark Multithreaded/Multiprocess Merge Sort
    multi_metrics = time_algorithm(
        multithreaded_merge_sort,
        data,
        trials=trials,
        num_threads=num_threads,
        executor_type=executor_type,
    )

    # Correctness Assertions
    seq_result = seq_metrics["result"]
    multi_result = multi_metrics["result"]
    is_identical = seq_result == multi_result
    assert is_identical, "Correctness assertion failed: Parallel output differs from Sequential output!"

    # Calculate Speedup and Parallel Efficiency
    t_seq = seq_metrics["mean_time"]
    t_multi = multi_metrics["mean_time"]
    speedup = (t_seq / t_multi) if t_multi > 0 else 1.0
    efficiency = (speedup / num_threads) if num_threads > 0 else 0.0

    # Clean records for tabular display
    comparison_records = [
        {
            "Algorithm": "Merge Sort (Single-Threaded)",
            "Input Size (n)": size,
            "Input Type": distribution,
            "Threads / Workers": 1,
            "Engine": "Sequential",
            "Mean Time (s)": round(t_seq, 5),
            "Min Time (s)": round(seq_metrics["min_time"], 5),
            "Max Time (s)": round(seq_metrics["max_time"], 5),
            "Std Dev (s)": round(seq_metrics["std_time"], 5),
            "Speedup": "1.00x (Baseline)",
        },
        {
            "Algorithm": f"Multithreaded Merge Sort ({executor_type.title()}Pool)",
            "Input Size (n)": size,
            "Input Type": distribution,
            "Threads / Workers": num_threads,
            "Engine": f"{executor_type.title()}PoolExecutor",
            "Mean Time (s)": round(t_multi, 5),
            "Min Time (s)": round(multi_metrics["min_time"], 5),
            "Max Time (s)": round(multi_metrics["max_time"], 5),
            "Std Dev (s)": round(multi_metrics["std_time"], 5),
            "Speedup": f"{speedup:.2f}x",
        },
    ]

    return {
        "size": size,
        "distribution": distribution,
        "num_threads": num_threads,
        "executor_type": executor_type,
        "trials": trials,
        "single_metrics": seq_metrics,
        "multi_metrics": multi_metrics,
        "speedup": speedup,
        "efficiency": efficiency,
        "is_correct": is_identical,
        "records_df": pd.DataFrame(comparison_records),
    }


def run_distribution_comparison_suite(
    size: int,
    num_threads: int,
    trials: int = 2,
    executor_type: str = "thread",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> pd.DataFrame:
    """
    Benchmarks both algorithms across all major input distributions.
    """
    rows = []
    total_steps = len(DISTRIBUTIONS)

    for idx, dist in enumerate(DISTRIBUTIONS):
        if progress_callback:
            progress_callback(
                (idx) / total_steps,
                f"Benchmarking distribution: {dist}...",
            )

        data = generate_data(size=size, distribution=dist, seed=100 + idx)

        # Single-threaded
        seq_res = time_algorithm(merge_sort, data, trials=trials)
        rows.append(
            {
                "Distribution": dist,
                "Algorithm": "Single-Threaded",
                "Mean Time (s)": seq_res["mean_time"],
                "Std Dev (s)": seq_res["std_time"],
                "Input Size": size,
                "Threads": 1,
            }
        )

        # Multithreaded
        multi_res = time_algorithm(
            multithreaded_merge_sort,
            data,
            trials=trials,
            num_threads=num_threads,
            executor_type=executor_type,
        )
        speedup = (
            (seq_res["mean_time"] / multi_res["mean_time"])
            if multi_res["mean_time"] > 0
            else 1.0
        )
        rows.append(
            {
                "Distribution": dist,
                "Algorithm": f"Multithreaded ({num_threads} workers)",
                "Mean Time (s)": multi_res["mean_time"],
                "Std Dev (s)": multi_res["std_time"],
                "Input Size": size,
                "Threads": num_threads,
                "Speedup": speedup,
            }
        )

    if progress_callback:
        progress_callback(1.0, "Distribution benchmark complete!")

    return pd.DataFrame(rows)


def run_scaling_benchmark_suite(
    sizes: List[int],
    distributions: List[str],
    num_threads: int,
    trials: int = 2,
    executor_type: str = "thread",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> pd.DataFrame:
    """
    Benchmarks both algorithms across varying input sizes n and distributions.
    Used for empirical asymptotic curve plotting vs theoretical O(n log n).
    """
    rows = []
    total_steps = len(sizes) * len(distributions)
    current_step = 0

    for size in sizes:
        for dist in distributions:
            current_step += 1
            if progress_callback:
                progress_callback(
                    current_step / total_steps,
                    f"Benchmarking n={size:,} ({dist})...",
                )

            data = generate_data(size=size, distribution=dist, seed=42)

            # Single-threaded
            t_seq = time_algorithm(merge_sort, data, trials=trials)["mean_time"]

            # Multithreaded
            t_multi = time_algorithm(
                multithreaded_merge_sort,
                data,
                trials=trials,
                num_threads=num_threads,
                executor_type=executor_type,
            )["mean_time"]

            speedup = (t_seq / t_multi) if t_multi > 0 else 1.0

            rows.append(
                {
                    "Input Size (n)": size,
                    "Distribution": dist,
                    "Single-Threaded Time (s)": t_seq,
                    "Multithreaded Time (s)": t_multi,
                    "Speedup": speedup,
                    "Theoretical N Log N (norm)": size * math.log2(max(2, size)),
                }
            )

    if progress_callback:
        progress_callback(1.0, "Scaling benchmark complete!")

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    print("Testing benchmark.py...")
    comp = run_comparison(size=5000, distribution="Random", num_threads=4, trials=2)
    print("Benchmark Comparison Sample:")
    print(comp["records_df"])
    print(f"Speedup: {comp['speedup']:.2f}x")
    print("benchmark.py test successful!")
