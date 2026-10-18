# ⚡ Merge Sort vs Multithreaded Merge Sort — Performance Comparison

**Course / Domain:** Design and Analysis of Algorithms (DAA)  
**Mini Project:** Parallel Divide-and-Conquer Performance & Correctness Study  
**Technologies:** Python 3.12, Streamlit, Plotly, NumPy, Pandas, `concurrent.futures`

---

## 📌 Executive Summary & Project Objective

The objective of this project is to implement, verify, and empirically compare the performance of **Classical Sequential Merge Sort** against a **Multithreaded / Multiprocess Merge Sort**.

Merge Sort is one of the foundational algorithms in computer science that exemplifies the **divide-and-conquer** paradigm. In this project, we explore:
1. **Algorithmic Correctness:** Validating mathematical invariants, stability, and identical output across implementations against Python's ground-truth Timsort.
2. **Parallel Decomposition:** Splitting datasets into $K$ chunks, sorting chunks concurrently across worker pools, and hierarchically combining them via pairwise parallel reduction.
3. **Hardware & Concurrency Constraints:** Analyzing the behavior of Python's **Global Interpreter Lock (GIL)** in `ThreadPoolExecutor` vs multi-core parallelism in `ProcessPoolExecutor`.
4. **Input Distribution Dynamics:** Evaluating Best Case (already sorted), Worst Case (reverse sorted), Nearly Sorted, and Random permutations to analyze constant-factor differences and branch prediction behavior.

---

## 🗂️ File Structure

```text
DAA/
├── app.py                # Streamlit UI with 4 interactive analysis tabs & Plotly graphs
├── algorithms.py         # Pure merge_sort, multithreaded_merge_sort, & reduction merge
├── benchmark.py          # Empirical timing harness, data generators, & comparison suites
├── requirements.txt      # Project dependencies (streamlit, plotly, pandas, numpy)
└── README.md             # Comprehensive report, theoretical proofs, & empirical findings
```

---

## 🚀 How to Setup and Run

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Clone / Navigate to Project Directory
```bash
cd "c:\Users\chait\OneDrive\Desktop\SEM 7\miniProject\DAA"
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit Interactive Dashboard
```bash
streamlit run app.py
```
The application will launch in your default web browser (usually at `http://localhost:8501`).

---

## 🧩 Architectural Implementation Details

### 1. Sequential Recursive Merge Sort (`merge_sort`)
Standard divide-and-conquer:
- Recursively divides array into two halves at `mid = len(arr) // 2`.
- Merges sorted halves using linear two-pointer `merge(left, right)`.
- Base case: `len(arr) <= 1`.

### 2. Multithreaded / Multiprocess Merge Sort (`multithreaded_merge_sort`)
Parallel divide-and-conquer architecture:
1. **Partitioning:** Given $N$ elements and $K$ workers (`os.cpu_count()`), partition into $K$ contiguous chunks of size $\approx \lceil N / K \rceil$.
2. **Phase 1 (Parallel Local Sort):** Submit all $K$ chunks to the execution pool (`ThreadPoolExecutor` or `ProcessPoolExecutor`). Each worker sorts its chunk independently using `merge_sort()`.
3. **Phase 2 (Parallel Pairwise Reduction Merge):**
   - In round 1, chunks $(C_0, C_1), (C_2, C_3), \dots$ are paired and merged concurrently.
   - Any odd leftover chunk carries over.
   - Repeats in $\lceil \log_2 K \rceil$ reduction rounds until a single sorted list remains.

```text
[ Input Array: Size N ]
           │
           ▼
┌──────────────────────────────────────────────────┐
│  Chunk Splitting: K Chunks of Size ~ (N / K)     │
└──────┬──────────────┬──────────────┬─────────────┘
       │              │              │
       ▼              ▼              ▼
  Worker 1       Worker 2       Worker K
[merge_sort]   [merge_sort]   [merge_sort]   <── Phase 1: Parallel Sorting
       │              │              │
       ▼              ▼              ▼
  Sorted C1      Sorted C2      Sorted Ck
       └──────┬───────┘              │
              ▼                      │
         Merge(C1, C2)               │       <── Phase 2: Pairwise Reduction
              └──────────────┬───────┘
                             ▼
                    Final Sorted Array
```

---

## 📐 Theoretical Complexity Analysis

| Metric | Sequential Merge Sort | Multithreaded Merge Sort ($K$ workers) | Notes |
| :--- | :--- | :--- | :--- |
| **Best Case Time** | $\Omega(n \log n)$ | $\Omega\left(\frac{n}{K} \log \frac{n}{K} + n \log K\right)$ | Classic merge sort does not short-circuit on sorted input |
| **Average Case Time** | $\Theta(n \log n)$ | $\Theta\left(\frac{n}{K} \log \frac{n}{K} + n \log K\right)$ | Uniform random permutations |
| **Worst Case Time** | $O(n \log n)$ | $O\left(\frac{n}{K} \log \frac{n}{K} + n \log K\right)$ | Reverse sorted input maximizes comparisons in merge |
| **Auxiliary Space** | $O(n)$ | $O(n + K)$ | Linear auxiliary buffer for merging sub-arrays |
| **Stability** | **Stable** ✅ | **Stable** ✅ | `<` vs `<=` maintains original relative order of duplicate keys |
| **Work Complexity ($T_1$)** | $O(n \log n)$ | $O(n \log n)$ | Total computational work across all processors |
| **Critical Path / Span ($T_\infty$)** | $O(n \log n)$ | $O\left(\frac{n}{K} \log \frac{n}{K} + n\right)$ | Longest sequential chain of dependencies |

### Master Theorem Derivation (Sequential)
$$T(n) = 2 \cdot T\left(\frac{n}{2}\right) + \Theta(n)$$
- Parameters: $a = 2$, $b = 2$, $f(n) = \Theta(n^1)$.
- Critical exponent: $\log_b a = \log_2 2 = 1$.
- Since $f(n) = \Theta(n^{\log_b a})$, by **Master Theorem Case 2**:
$$T(n) = \Theta(n \log n)$$

### Amdahl's Law & Parallel Speedup Limits
Amdahl's Law defines the theoretical speedup $S(K)$ achievable using $K$ parallel processors:
$$S_{\text{latency}}(K) = \frac{1}{(1 - p) + \frac{p}{K}}$$
Where:
- $p$ is the parallelizable fraction (sorting individual chunks: $O(\frac{n}{K} \log \frac{n}{K})$).
- $(1 - p)$ is the inherently sequential portion (the final levels of pairwise reduction merging where all $n$ elements must be processed sequentially: $O(n)$).

Even with an infinite number of processors ($K \to \infty$), the maximum speedup is strictly bounded by:
$$S_{\max} = \frac{1}{1 - p}$$

---

## ⚖️ Best Case vs. Worst Case Dynamics

1. **Best Case (Already / Nearly Sorted):**
   - *Time Complexity:* $\Theta(n \log n)$.
   - *Behavior:* Unlike Insertion Sort or Python's native Timsort (which can detect runs in $O(n)$), classic merge sort decomposes unconditionally down to single elements.
   - *Constant-Factor Advantage:* During linear merging, one sub-array's elements are consistently smaller than the other. This allows the CPU hardware branch predictor to achieve near-100% prediction accuracy, minimizing instruction pipeline flushes.

2. **Worst Case (Reverse Sorted):**
   - *Time Complexity:* $\Theta(n \log n)$.
   - *Behavior:* Elements from the left and right halves strictly alternate comparisons during each merge step.
   - *Constant-Factor Disadvantage:* Maximizes total comparison count ($(n - 1)$ comparisons for merging two arrays of size $n/2$) and causes high branch predictor variance.

---

## 🧵 The Python GIL & Concurrency Findings

### Why is Multithreading Limited in CPython?
CPython features a **Global Interpreter Lock (GIL)**, a mutual-exclusion lock designed to prevent concurrent native threads from executing Python bytecode simultaneously.

- **`ThreadPoolExecutor` for CPU-Bound Sorting:**
  Because sorting involves continuous Python object comparisons and list manipulations, threads constantly contest for the GIL. The OS rapidly context-switches between threads, introducing thread synchronization overhead without true multi-core speedup. For sizes $n < 25,000$, single-threaded execution is frequently faster than `ThreadPoolExecutor`.
- **`ProcessPoolExecutor` (True Multi-Core):**
  Spawns independent OS processes, each with its own isolated Python interpreter and independent GIL. This bypasses the lock completely. However, on Windows, process spawning (`spawn` model) and inter-process communication (IPC pickling of large lists) introduce substantial constant overhead. True speedup emerges when $n > 30,000 - 50,000$, where compute time outweighs IPC serialization.
- **Compiled Languages (C/C++, Go, Rust):**
  In native languages with thread primitives (e.g. OpenMP, POSIX threads, Go goroutines), multithreaded merge sort operates directly on raw memory without GIL locks or object boxing, achieving near-linear $K \times$ speedup.

---

## 🖥️ Streamlit Web App Features

The application is structured into four dedicated tabs:

1. **🧪 Verify Correctness:**
   - Input custom comma-separated lists or choose from built-in presets (negatives, duplicates, reversed, sorted).
   - Generates random test arrays on demand.
   - Executes `merge_sort`, `multithreaded_merge_sort`, and Python's built-in `sorted()`.
   - Programmatically verifies mathematical equivalence via assertions:
     `assert seq_sorted == multi_sorted == python_sorted`.
   - Side-by-side comparison tables and interactive Plotly order visualizers.

2. **📊 Benchmark & Speedup:**
   - Sliders for input size $n$ (100 to 100,000), worker count $K$, input distribution, and trial count.
   - Grouped interactive Plotly bar charts comparing execution times with standard deviation error bars.
   - High-precision microsecond measurements via `time.perf_counter()`.
   - Metric cards: Single-Threaded Time, Multithreaded Time, Speedup ($S = T_{seq} / T_{par}$), Efficiency ($E = S / K$).
   - Interactive data table and **"Download Results as CSV"** button (`st.download_button`).

3. **⚖️ Best vs Worst Case:**
   - Compares Sorted vs Nearly Sorted vs Random vs Reverse Sorted.
   - Grouped distribution bar chart.
   - Multi-point asymptotic scaling line chart ($n = 500 \to 50,000$) illustrating empirical $O(n \log n)$ curvature.
   - Constant-factor and overhead breakdown notes.

4. **🧠 Complexity & Theoretical Analysis:**
   - Complete Big-O comparison table (Best, Average, Worst, Auxiliary Space, Stability, Work $T_1$, Span $T_\infty$).
   - Master Theorem Case 2 proof derivation.
   - Amdahl's Law mathematical explanation.
   - Full architectural ASCII flowchart illustrating parallel chunking and pairwise reduction merge.

---

## 🔬 Key Empirical Findings & Conclusions

1. **Threshold Effect:**
   Parallel sorting incurs overhead: chunk slicing $O(n)$, thread pool dispatch $O(K)$, and pairwise reduction merging $O(n \log K)$. For small input sizes ($n < 10,000$), sequential merge sort consistently outperforms multithreading.
2. **Speedup Emergence:**
   Speedup becomes evident as $n$ scales into tens of thousands of elements, where the $O(n \log n)$ sorting workload dominates fixed scheduling overheads.
3. **GIL Impact:**
   Python's GIL restricts `ThreadPoolExecutor` to cooperative time-slicing for CPU-bound tasks. To observe true hardware-level multi-core acceleration in Python, developers must use `ProcessPoolExecutor`, native extensions (Cython/C++), or numerical runtimes (NumPy/Numba) that release the GIL.
4. **Distribution Invariance:**
   Classic merge sort maintains consistent asymptotic $O(n \log n)$ scaling across all four distributions, with sorted data holding a modest constant-factor advantage due to cache locality and CPU branch prediction.

---

## 👨‍💻 Author & Project Info
- **Project:** Merge Sort vs Multithreaded Merge Sort Analyzer
- **Domain:** Design and Analysis of Algorithms (DAA)
- **Course Level:** Semester 7 Mini Project
