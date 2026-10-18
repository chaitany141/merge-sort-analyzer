"""
algorithms.py
=============
Implementation of standard recursive Merge Sort and Multithreaded/Multiprocess
Merge Sort using Python's concurrent.futures module.

Theoretical Foundations:
-----------------------
1. Standard Merge Sort:
   - Divide-and-conquer paradigm: T(n) = 2*T(n/2) + O(n).
   - Time Complexity: O(n log n) in Best, Average, and Worst cases.
   - Space Complexity: O(n) auxiliary memory.
   - Stability: Preserved (<= comparator guarantees relative order).

2. Parallel Merge Sort:
   - Partition array into K chunks (where K = num_threads).
   - Parallel Sort Phase: Each chunk sorted concurrently via merge_sort.
   - Parallel Merge Phase: Pairwise tree-reduction merge of sorted chunks.
   - In CPython, ThreadPoolExecutor is bounded by the Global Interpreter Lock (GIL)
     for CPU-bound operations. ProcessPoolExecutor bypasses the GIL by spawning
     independent worker processes with isolated Python runtimes.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, List, Sequence, Tuple


def merge(left: Sequence[Any], right: Sequence[Any]) -> List[Any]:
    """
    Merges two pre-sorted sequences into a single sorted list.

    Parameters:
        left: First sorted sequence.
        right: Second sorted sequence.

    Returns:
        A new sorted list containing all elements from left and right.
    """
    merged: List[Any] = []
    i = 0
    j = 0
    len_left = len(left)
    len_right = len(right)

    # Linear two-pointer traversal
    while i < len_left and j < len_right:
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    # Append any remaining tail elements
    if i < len_left:
        merged.extend(left[i:])
    if j < len_right:
        merged.extend(right[j:])

    return merged


def merge_sort(arr: Sequence[Any]) -> List[Any]:
    """
    Standard recursive Merge Sort algorithm.

    Recurrence Relation:
        T(n) = 2 * T(n/2) + O(n) -> Theta(n log n) by Master Theorem (Case 2).

    Parameters:
        arr: Sequence of comparable elements.

    Returns:
        A new sorted list.
    """
    n = len(arr)
    if n <= 1:
        return list(arr)

    mid = n // 2
    left_sorted = merge_sort(arr[:mid])
    right_sorted = merge_sort(arr[mid:])

    return merge(left_sorted, right_sorted)


def _merge_pair_helper(pair: Tuple[List[Any], List[Any]]) -> List[Any]:
    """Helper function to merge a 2-tuple of sorted lists (pickleable for multiprocessing)."""
    return merge(pair[0], pair[1])


def parallel_pairwise_merge(
    sorted_chunks: List[List[Any]], executor: Executor
) -> List[Any]:
    """
    Merges a list of sorted chunks pairwise using parallel reduction.

    In each round:
        - Pairs of consecutive chunks (chunk[2i], chunk[2i+1]) are merged concurrently.
        - If the count of chunks is odd, the last chunk is carried forward to the next round.
        - This tree reduction completes in ceil(log2(K)) rounds.

    Parameters:
        sorted_chunks: List of sorted chunks.
        executor: concurrent.futures.Executor instance (ThreadPoolExecutor or ProcessPoolExecutor).

    Returns:
        The final combined sorted list.
    """
    if not sorted_chunks:
        return []

    current_chunks = sorted_chunks

    while len(current_chunks) > 1:
        pairs: List[Tuple[List[Any], List[Any]]] = []
        carry_over: List[List[Any]] = []

        i = 0
        total = len(current_chunks)
        while i < total:
            if i + 1 < total:
                pairs.append((current_chunks[i], current_chunks[i + 1]))
                i += 2
            else:
                carry_over.append(current_chunks[i])
                i += 1

        # Submit pairwise merges in parallel
        if pairs:
            # When only 1 pair exists, run directly to save executor overhead
            if len(pairs) == 1 and not carry_over:
                return merge(pairs[0][0], pairs[0][1])

            futures = [executor.submit(_merge_pair_helper, pair) for pair in pairs]
            merged_results = [f.result() for f in futures]
        else:
            merged_results = []

        current_chunks = merged_results + carry_over

    return current_chunks[0] if current_chunks else []


def multithreaded_merge_sort(
    arr: Sequence[Any],
    num_threads: int | None = None,
    executor_type: str = "thread",
) -> List[Any]:
    """
    Parallel Merge Sort utilizing Python's concurrent.futures execution pools.

    Algorithm Workflow:
    -------------------
    1. Determine worker count K = min(num_threads, len(arr)).
    2. Partition input array into K roughly equal-sized sub-chunks.
    3. Concurrently sort each sub-chunk using standard merge_sort.
    4. Hierarchically merge the sorted sub-chunks pairwise using parallel reduction.

    Concurrency Modes:
    ------------------
    - "thread": Uses ThreadPoolExecutor. Subject to CPython Global Interpreter Lock (GIL).
                Useful for demonstrating thread pool synchronization, overhead, and GIL constraints.
    - "process": Uses ProcessPoolExecutor. Bypasses the GIL across multiple CPU cores.
                 Demonstrates true parallel speedup for larger input sizes.

    Parameters:
        arr: Sequence of elements to sort.
        num_threads: Number of worker threads/processes. Defaults to os.cpu_count().
        executor_type: 'thread' for ThreadPoolExecutor or 'process' for ProcessPoolExecutor.

    Returns:
        A new sorted list.
    """
    n = len(arr)
    if n <= 1:
        return list(arr)

    # Resolve default thread count
    if num_threads is None or num_threads < 1:
        num_threads = os.cpu_count() or 4

    # If single thread requested or array too small for parallelism, run sequential
    if num_threads == 1 or n < 2 * num_threads:
        return merge_sort(arr)

    # Bound chunks by array size
    k = min(num_threads, n)
    chunk_size = math.ceil(n / k)
    chunks = [list(arr[i : i + chunk_size]) for i in range(0, n, chunk_size)]

    # Select execution pool
    is_process = executor_type.lower() == "process"
    executor_cls = ProcessPoolExecutor if is_process else ThreadPoolExecutor

    with executor_cls(max_workers=k) as executor:
        # Phase 1: Parallel chunk sorting
        sort_futures = [executor.submit(merge_sort, chunk) for chunk in chunks]
        sorted_chunks = [f.result() for f in sort_futures]

        # Phase 2: Parallel pairwise tree reduction merge
        final_sorted = parallel_pairwise_merge(sorted_chunks, executor)

    return final_sorted


if __name__ == "__main__":
    # Self-test on module execution
    import random

    test_data = [random.randint(-1000, 1000) for _ in range(500)]
    expected = sorted(test_data)

    seq_res = merge_sort(test_data)
    assert seq_res == expected, "Sequential merge sort failed correctness test!"

    thread_res = multithreaded_merge_sort(test_data, num_threads=4, executor_type="thread")
    assert thread_res == expected, "Multithreaded merge sort failed correctness test!"

    process_res = multithreaded_merge_sort(test_data, num_threads=4, executor_type="process")
    assert process_res == expected, "Multiprocess merge sort failed correctness test!"

    print("All algorithms.py self-tests passed successfully!")
