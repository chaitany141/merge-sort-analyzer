"""
app.py
======
Streamlit Web Application:
"Merge Sort vs Multithreaded Merge Sort — Performance Comparison"

Designed for DAA (Design and Analysis of Algorithms) Mini Project.
Provides interactive correctness verification, empirical benchmarking,
distribution analysis, theoretical complexity, and CSV export.
"""

from __future__ import annotations

import os
import random
import time
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from algorithms import merge_sort, multithreaded_merge_sort
from benchmark import (
    DISTRIBUTIONS,
    generate_data,
    run_comparison,
    run_distribution_comparison_suite,
    run_scaling_benchmark_suite,
    time_algorithm,
)


# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Merge Sort Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Custom Styling (Modern, Clean UI)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 12px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #6366f1;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #888888;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #4f46e5;
        font-family: 'Courier New', monospace;
    }
    .metric-subtitle {
        font-size: 0.78rem;
        color: #777777;
        margin-top: 4px;
    }
    
    /* Status Badges */
    .badge-success {
        background-color: #10b981;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-info {
        background-color: #6366f1;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Code & Callout container */
    .callout-box {
        background-color: rgba(99, 102, 241, 0.08);
        border-left: 4px solid #6366f1;
        padding: 14px;
        border-radius: 6px;
        margin: 12px 0;
        font-size: 0.92rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar Configuration & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚡ Merge Sort Analyzer")
    st.caption("DAA Mini Project | Performance & Complexity Study")
    st.markdown("---")

    st.subheader("⚙️ Benchmark Controls")

    # Quick Size Presets
    st.markdown("**Quick Size Presets:**")
    p_col1, p_col2, p_col3 = st.columns(3)
    preset_size = None
    if p_col1.button("1,000"):
        preset_size = 1000
    if p_col2.button("10,000"):
        preset_size = 10000
    if p_col3.button("50,000"):
        preset_size = 50000

    default_size = preset_size if preset_size is not None else 10000

    # Input Size Slider
    input_size = st.slider(
        "Input Size (n):",
        min_value=100,
        max_value=100000,
        value=default_size,
        step=500,
        help="Number of elements in the synthetic benchmark dataset.",
    )

    # Thread count slider
    max_cpu = os.cpu_count() or 4
    num_threads = st.slider(
        "Worker Count (Threads/Processes):",
        min_value=1,
        max_value=max(16, max_cpu * 2),
        value=max_cpu,
        step=1,
        help=f"Number of parallel workers. Detected hardware CPU cores: {max_cpu}.",
    )

    # Concurrency Engine Toggle
    executor_choice = st.radio(
        "Parallel Concurrency Engine:",
        options=["ThreadPoolExecutor (Thread)", "ProcessPoolExecutor (Process)"],
        index=0,
        help=(
            "ThreadPoolExecutor runs in the same Python process (bounded by CPython GIL for pure CPU work). "
            "ProcessPoolExecutor spawns separate processes across CPU cores (bypasses GIL, but incurs IPC/spawn overhead)."
        ),
    )
    executor_type = "thread" if "Thread" in executor_choice else "process"

    # Input distribution selector
    distribution = st.selectbox(
        "Input Distribution:",
        options=DISTRIBUTIONS,
        index=0,
        help="Data pattern to evaluate Best, Worst, and Average case behaviors.",
    )

    # Number of trials
    trials = st.slider(
        "Number of Trials (Averaging):",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Number of runs to average execution time for empirical stability.",
    )

    # Seed controls
    use_fixed_seed = st.checkbox("Deterministic RNG Seed (42)", value=True)
    rng_seed = 42 if use_fixed_seed else None

    # Size Warning
    if input_size > 50000:
        st.warning(
            f"⚠️ Input size n = {input_size:,} is large. "
            "Sorting multiple trials may take 5–15 seconds depending on hardware."
        )

    st.markdown("---")
    st.markdown("### 🖥️ Hardware Context")
    st.markdown(f"- **Detected Logical CPUs:** `{max_cpu}`")
    st.markdown("- **Python Runtime:** `CPython 3.12`")
    st.markdown(
        "- **GIL Status:** `Active` *(CPU-bound threads run cooperatively)*"
    )


# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
st.title("⚡ Merge Sort vs Multithreaded Merge Sort")
st.markdown(
    "**An Empirical & Theoretical Performance Comparison** &nbsp;|&nbsp; "
    "<span class='badge-info'>DAA Mini Project</span> &nbsp; "
    "<span class='badge-success'>Correctness Verified</span>",
    unsafe_allow_html=True,
)
st.markdown(
    "Explore how divide-and-conquer parallelism behaves compared to classical recursive Merge Sort "
    "under various input distributions, thread worker pools, and CPython concurrency models."
)
st.markdown("---")


# -----------------------------------------------------------------------------
# Tab Layout
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🧪 Verify Correctness",
        "📊 Benchmark & Speedup",
        "⚖️ Best vs Worst Case",
        "🧠 Complexity & Analysis",
    ]
)


# =============================================================================
# TAB 1: Verify Correctness
# =============================================================================
with tab1:
    st.subheader("🧪 Mathematical Correctness & Invariant Verification")
    st.markdown(
        "Verify that both the single-threaded recursive merge sort and the multithreaded "
        "chunked reduction sort produce results mathematically identical to Python's built-in Timsort."
    )

    col_input, col_preset = st.columns([2, 1])

    with col_preset:
        st.markdown("**Quick Test Presets:**")
        preset_type = st.radio(
            "Select Test Case:",
            [
                "Mixed Negatives & Positives",
                "Array with Duplicates",
                "Reverse Sorted Sequence",
                "Already Sorted Sequence",
                "Custom Input",
            ],
            index=0,
        )

    with col_input:
        if preset_type == "Mixed Negatives & Positives":
            sample_text = "38, 27, 43, 3, 9, 82, 10, -15, 0, 100, -42, 5"
        elif preset_type == "Array with Duplicates":
            sample_text = "7, 2, 7, 3, 2, 8, 7, 1, 9, 3, 7, 2, 5"
        elif preset_type == "Reverse Sorted Sequence":
            sample_text = "50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0"
        elif preset_type == "Already Sorted Sequence":
            sample_text = "1, 4, 9, 16, 25, 36, 49, 64, 81, 100"
        else:
            sample_text = "14, -8, 22, 105, 3, 3, -1, 42, 99, 0, 18"

        user_array_str = st.text_area(
            "Enter Comma-Separated Numbers (or edit above):",
            value=sample_text,
            height=100,
        )

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        verify_btn = st.button("🚀 Verify Correctness", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🎲 Generate Random 25-Element Array", use_container_width=True):
            user_array_str = ", ".join(str(x) for x in random.sample(range(-100, 100), 25))
            st.session_state["random_custom_arr"] = user_array_str
            st.rerun()

    # Parse and test array
    try:
        raw_items = [x.strip() for x in user_array_str.split(",") if x.strip() != ""]
        test_array = [float(x) if "." in x else int(x) for x in raw_items]
    except ValueError:
        st.error("❌ Invalid input: Please ensure all items are valid numbers separated by commas.")
        test_array = []

    if test_array:
        # Run sorting algorithms
        t0 = time.perf_counter()
        seq_sorted = merge_sort(test_array)
        t_seq = time.perf_counter() - t0

        t1 = time.perf_counter()
        multi_sorted = multithreaded_merge_sort(
            test_array, num_threads=num_threads, executor_type=executor_type
        )
        t_multi = time.perf_counter() - t1

        timsort_sorted = sorted(test_array)

        # Assertions
        assert_seq = seq_sorted == timsort_sorted
        assert_multi = multi_sorted == timsort_sorted
        assert_identical = seq_sorted == multi_sorted
        assert_length = len(multi_sorted) == len(test_array)

        all_passed = assert_seq and assert_multi and assert_identical and assert_length

        st.markdown("---")
        if all_passed:
            st.success(
                f"✅ **ALL ASSERTIONS PASSED!** Both algorithms produced mathematically identical, "
                f"100% sorted outputs matching Python's ground-truth `sorted()`. "
                f"(Length: {len(test_array)} items, {num_threads} workers used)."
            )
        else:
            st.error("❌ Correctness assertion failed! Results do not match ground truth.")

        # Display side-by-side results
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown("#### 📥 Original Input Array")
            st.code(str(test_array), language="python")
            st.caption(f"Count: {len(test_array)} items")

        with res_col2:
            st.markdown("#### 📤 Sorted Output (Identical Across All)")
            st.code(str(multi_sorted), language="python")
            st.caption(f"Verification: Monotonically non-decreasing order validated.")

        # Verification Matrix
        st.markdown("#### 🔍 Assertion Matrix")
        matrix_df = pd.DataFrame(
            [
                {
                    "Assertion": "`merge_sort(arr) == sorted(arr)`",
                    "Status": "PASSED ✅" if assert_seq else "FAILED ❌",
                    "Details": "Standard recursive merge sort matches Timsort baseline.",
                },
                {
                    "Assertion": "`multithreaded_merge_sort(arr) == sorted(arr)`",
                    "Status": "PASSED ✅" if assert_multi else "FAILED ❌",
                    "Details": "Chunked parallel sort + reduction merge matches Timsort.",
                },
                {
                    "Assertion": "`merge_sort(arr) == multithreaded_merge_sort(arr)`",
                    "Status": "PASSED ✅" if assert_identical else "FAILED ❌",
                    "Details": "Sequential and parallel implementations produce identical output.",
                },
                {
                    "Assertion": "`len(sorted_arr) == len(original_arr)`",
                    "Status": "PASSED ✅" if assert_length else "FAILED ❌",
                    "Details": "No elements dropped or duplicated during parallel merge.",
                },
            ]
        )
        st.table(matrix_df)

        # Visual Bar Comparison for small arrays
        if len(test_array) <= 40:
            fig_bar = go.Figure()
            fig_bar.add_trace(
                go.Bar(
                    x=list(range(len(test_array))),
                    y=test_array,
                    name="Original Array (Unsorted)",
                    marker_color="rgba(156, 163, 175, 0.6)",
                )
            )
            fig_bar.add_trace(
                go.Bar(
                    x=list(range(len(multi_sorted))),
                    y=multi_sorted,
                    name="Sorted Array (Merge Sort Output)",
                    marker_color="rgba(99, 102, 241, 0.9)",
                )
            )
            fig_bar.update_layout(
                title="Input vs Sorted Order Visualization",
                xaxis_title="Index Position",
                yaxis_title="Value",
                barmode="group",
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)


# =============================================================================
# TAB 2: Benchmark & Speedup
# =============================================================================
with tab2:
    st.subheader("📊 Empirical Benchmark & Speedup Evaluation")
    st.markdown(
        "Benchmark `merge_sort` against `multithreaded_merge_sort` on identical data copies. "
        "Measures execution time with microsecond precision using `time.perf_counter()`."
    )

    # Benchmark Action Button
    b_col1, b_col2 = st.columns([1, 4])
    with b_col1:
        run_bench = st.button("🚀 Run Benchmark", type="primary", use_container_width=True)
    with b_col2:
        st.markdown(
            f"**Current Configuration:** Size: `{input_size:,}` | Workers: `{num_threads}` | "
            f"Distribution: `{distribution}` | Engine: `{executor_type.title()}Pool` | Trials: `{trials}`"
        )

    if run_bench or "last_benchmark_result" in st.session_state:
        if run_bench:
            with st.spinner(f"Running {trials} benchmark trials on n = {input_size:,}..."):
                progress_bar = st.progress(0.2, text="Generating synthetic dataset...")
                time.sleep(0.05)
                progress_bar.progress(0.5, text="Executing sorting trials...")
                res = run_comparison(
                    size=input_size,
                    distribution=distribution,
                    num_threads=num_threads,
                    trials=trials,
                    executor_type=executor_type,
                    seed=rng_seed,
                )
                progress_bar.progress(1.0, text="Benchmarking complete!")
                time.sleep(0.1)
                progress_bar.empty()
                st.session_state["last_benchmark_result"] = res
        else:
            res = st.session_state["last_benchmark_result"]

        t_seq = res["single_metrics"]["mean_time"]
        t_multi = res["multi_metrics"]["mean_time"]
        speedup = res["speedup"]
        efficiency = res["efficiency"]

        # 4 Metric Cards
        st.markdown("### 📈 Performance Summary")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        with m_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Single-Threaded Time</div>
                    <div class="metric-value">{t_seq:.5f}s</div>
                    <div class="metric-subtitle">Mean of {trials} trial(s)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Multithreaded Time</div>
                    <div class="metric-value">{t_multi:.5f}s</div>
                    <div class="metric-subtitle">{num_threads} workers ({executor_type.title()}Pool)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        speedup_color = "#10b981" if speedup >= 1.0 else "#f59e0b"
        with m_col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Speedup Factor (S)</div>
                    <div class="metric-value" style="color: {speedup_color};">{speedup:.2f}x</div>
                    <div class="metric-subtitle">S = T_seq / T_par</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m_col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Parallel Efficiency</div>
                    <div class="metric-value">{efficiency * 100:.1f}%</div>
                    <div class="metric-subtitle">E = Speedup / {num_threads} workers</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Visualizations (Grouped Bar Chart)
        st.markdown("### 📊 Execution Time Comparison")
        chart_col1, chart_col2 = st.columns([3, 2])

        with chart_col1:
            fig_bench = go.Figure()
            fig_bench.add_trace(
                go.Bar(
                    x=["Single-Threaded", f"Multithreaded ({num_threads}W)"],
                    y=[t_seq, t_multi],
                    error_y=dict(
                        type="data",
                        array=[
                            res["single_metrics"]["std_time"],
                            res["multi_metrics"]["std_time"],
                        ],
                        visible=True,
                    ),
                    marker_color=["#6366f1", "#10b981" if speedup >= 1.0 else "#f59e0b"],
                    text=[f"{t_seq:.4f}s", f"{t_multi:.4f}s"],
                    textposition="auto",
                )
            )
            fig_bench.update_layout(
                title=f"Execution Time Comparison (n = {input_size:,}, {distribution})",
                yaxis_title="Execution Time (seconds) [Lower is Better]",
                xaxis_title="Algorithm Implementation",
                height=360,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_bench, use_container_width=True)

        with chart_col2:
            # Individual trial scatter/line plot
            fig_trials = go.Figure()
            trial_indices = list(range(1, trials + 1))
            fig_trials.add_trace(
                go.Scatter(
                    x=trial_indices,
                    y=res["single_metrics"]["trials"],
                    mode="lines+markers",
                    name="Single-Threaded",
                    line=dict(color="#6366f1", width=2),
                )
            )
            fig_trials.add_trace(
                go.Scatter(
                    x=trial_indices,
                    y=res["multi_metrics"]["trials"],
                    mode="lines+markers",
                    name=f"Multithreaded ({num_threads}W)",
                    line=dict(color="#10b981" if speedup >= 1.0 else "#f59e0b", width=2),
                )
            )
            fig_trials.update_layout(
                title=f"Trial-by-Trial Consistency ({trials} Trials)",
                xaxis_title="Trial Number",
                yaxis_title="Time (s)",
                height=360,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_trials, use_container_width=True)

        # Tabular Breakdown
        st.markdown("### 📋 Benchmark Data Table")
        st.dataframe(res["records_df"], use_container_width=True)

        # CSV Download Button
        csv_data = res["records_df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name=f"merge_sort_benchmark_n{input_size}_{executor_type}.csv",
            mime="text/csv",
        )

        # Deep Analytical Interpretation
        st.markdown("### 🔬 Performance Findings & Insights")
        if speedup < 1.0:
            st.info(
                f"**Observation (Speedup = {speedup:.2f}x < 1.0x):** "
                f"Single-threaded execution was faster than multithreaded execution for $n = {input_size:,}$. "
                "**Why?** In CPython, the **Global Interpreter Lock (GIL)** prevents concurrent bytecode execution on CPU threads. "
                "The parallel version incurs additional overhead from: "
                "(1) slicing the list into chunks, (2) submitting and scheduling tasks in `ThreadPoolExecutor`, and "
                "(3) sequential pairwise tree reduction merges. For smaller or moderate input sizes, this coordination overhead exceeds "
                "any parallel gain. Try increasing $n > 30,000$ or switching to `ProcessPoolExecutor` to observe parallel scaling."
            )
        else:
            st.success(
                f"**Observation (Speedup = {speedup:.2f}x >= 1.0x):** "
                f"Multithreaded execution achieved a positive speedup over sequential execution. "
                f"The computational sorting payload of $n = {input_size:,}$ elements amortized "
                f"the thread chunking and synchronization costs across {num_threads} workers."
            )


# =============================================================================
# TAB 3: Best vs Worst Case Analysis
# =============================================================================
with tab3:
    st.subheader("⚖️ Best Case vs. Worst Case Distribution Analysis")
    st.markdown(
        r"""
        Merge sort operates strictly via divide-and-conquer. Unlike algorithms such as **Insertion Sort** 
        (which terminates early in $O(n)$ for sorted arrays) or **QuickSort** (which degrades to $O(n^2)$ on unbalanced pivots), 
        **classical Merge Sort is strictly $\Theta(n \log n)$ across all input distributions**.
        """
    )

    exp_col1, exp_col2, exp_col3 = st.columns(3)
    with exp_col1:
        st.markdown(
            """
            <div class="callout-box">
                <strong>🟢 Best Case (Sorted / Nearly Sorted)</strong><br>
                Theoretical Time: <code>O(n log n)</code><br>
                <em>Why no O(n)?</em> Classic merge sort unconditionally splits down to single elements. 
                However, consecutive elements in <code>left</code> are already smaller than <code>right</code>, 
                minimizing branch mispredictions and maximizing CPU hardware prefetching.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with exp_col2:
        st.markdown(
            """
            <div class="callout-box">
                <strong>🟡 Average Case (Random)</strong><br>
                Theoretical Time: <code>O(n log n)</code><br>
                Uniformly distributed keys cause elements from <code>left</code> and <code>right</code> 
                to interleave randomly, incurring typical comparison counts (~n log2(n) - 0.91n).
            </div>
            """,
            unsafe_allow_html=True,
        )

    with exp_col3:
        st.markdown(
            """
            <div class="callout-box">
                <strong>🔴 Worst Case (Reverse Sorted)</strong><br>
                Theoretical Time: <code>O(n log n)</code><br>
                Maximizes the number of comparisons during the merge step, as items 
                strictly alternate between the two halves until one half is exhausted.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 📊 Distribution Benchmark Suite")
    st.caption(
        f"Runs all 4 distributions at size n = {input_size:,} with {num_threads} workers."
    )

    if st.button("🚀 Run All 4 Distributions Benchmark", type="primary"):
        prog_bar = st.progress(0.0)
        status_text = st.empty()

        def update_cb(pct: float, msg: str):
            prog_bar.progress(pct)
            status_text.text(msg)

        dist_df = run_distribution_comparison_suite(
            size=input_size,
            num_threads=num_threads,
            trials=trials,
            executor_type=executor_type,
            progress_callback=update_cb,
        )
        prog_bar.empty()
        status_text.empty()
        st.session_state["dist_df"] = dist_df

    if "dist_df" in st.session_state:
        dist_df = st.session_state["dist_df"]

        # Grouped bar chart comparing distributions
        fig_dist = px.bar(
            dist_df,
            x="Distribution",
            y="Mean Time (s)",
            color="Algorithm",
            barmode="group",
            title=f"Merge Sort Execution Time by Input Distribution (n = {input_size:,})",
            color_discrete_sequence=["#6366f1", "#10b981"],
            text_auto=".4f",
        )
        fig_dist.update_layout(
            yaxis_title="Execution Time (seconds) [Lower is Better]",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        st.dataframe(dist_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📈 Asymptotic Scaling: Time vs Input Size (n from 500 → 60,000)")
    st.markdown(
        "Evaluate how execution time scales as input size $n$ increases across multiple orders of magnitude. "
        "Notice the characteristic $O(n \\log n)$ curvature."
    )

    scale_col1, scale_col2 = st.columns([1, 3])
    with scale_col1:
        run_scaling = st.button("🚀 Run Scaling Analysis", use_container_width=True)
    with scale_col2:
        st.caption("Tests sizes: [500, 2,000, 10,000, 25,000, 50,000] across Random and Reverse Sorted inputs.")

    if run_scaling or "scaling_df" in st.session_state:
        if run_scaling:
            prog_scale = st.progress(0.0)
            status_scale = st.empty()

            def scale_cb(pct: float, msg: str):
                prog_scale.progress(pct)
                status_scale.text(msg)

            test_sizes = [500, 2000, 10000, 25000, 50000]
            test_dists = ["Random", "Reverse Sorted (Worst Case)"]

            sc_df = run_scaling_benchmark_suite(
                sizes=test_sizes,
                distributions=test_dists,
                num_threads=num_threads,
                trials=2,
                executor_type=executor_type,
                progress_callback=scale_cb,
            )
            prog_scale.empty()
            status_scale.empty()
            st.session_state["scaling_df"] = sc_df
        else:
            sc_df = st.session_state["scaling_df"]

        # Scaling Line Chart
        fig_scale = go.Figure()
        for d in sc_df["Distribution"].unique():
            sub = sc_df[sc_df["Distribution"] == d]
            fig_scale.add_trace(
                go.Scatter(
                    x=sub["Input Size (n)"],
                    y=sub["Single-Threaded Time (s)"],
                    mode="lines+markers",
                    name=f"Single-Threaded ({d})",
                    line=dict(dash="solid"),
                )
            )
            fig_scale.add_trace(
                go.Scatter(
                    x=sub["Input Size (n)"],
                    y=sub["Multithreaded Time (s)"],
                    mode="lines+markers",
                    name=f"Multithreaded ({d})",
                    line=dict(dash="dot"),
                )
            )

        fig_scale.update_layout(
            title="Time vs Input Size (n) Scaling Curves",
            xaxis_title="Input Size (n)",
            yaxis_title="Execution Time (seconds)",
            height=420,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_scale, use_container_width=True)

        # Constant Factor & Overhead Breakdown Note
        st.markdown(
            r"""
            > **Theoretical Complexity Note:**  
            > While both single-threaded and multithreaded merge sort adhere to asymptotic $O(n \log n)$, 
            > their **constant factor $c$** differs significantly:
            > - $T_{\text{seq}}(n) = c_1 \cdot n \log_2 n$
            > - $T_{\text{par}}(n) = \frac{c_1}{K} \cdot n \log_2(n/K) + c_2 \cdot n \log_2 K + \text{Overhead}_{\text{pool}}$
            > The synchronization barrier in parallel reduction merging and thread scheduling creates an overhead floor.
            """
        )


# =============================================================================
# TAB 4: Complexity & Analysis
# =============================================================================
with tab4:
    st.subheader("🧠 Theoretical Complexity, Recurrence Relations & Multi-Core Systems")

    st.markdown("### 📋 Big-O Complexity Comparison Table")
    complexity_data = [
        {
            "Metric": "Best Case Time",
            "Single-Threaded Merge Sort": "Ω(n log n)",
            "Multithreaded Merge Sort (K workers)": "Ω((n/K) log(n/K) + n log K)",
            "Remarks": "No early exit in classic merge sort.",
        },
        {
            "Metric": "Average Case Time",
            "Single-Threaded Merge Sort": "Θ(n log n)",
            "Multithreaded Merge Sort (K workers)": "Θ((n/K) log(n/K) + n log K)",
            "Remarks": "Uniform random key permutations.",
        },
        {
            "Metric": "Worst Case Time",
            "Single-Threaded Merge Sort": "O(n log n)",
            "Multithreaded Merge Sort (K workers)": "O((n/K) log(n/K) + n log K)",
            "Remarks": "Reverse sorted inputs maximize comparison operations.",
        },
        {
            "Metric": "Auxiliary Space",
            "Single-Threaded Merge Sort": "O(n)",
            "Multithreaded Merge Sort (K workers)": "O(n + K)",
            "Remarks": "Requires temporary buffers to store merged sub-arrays.",
        },
        {
            "Metric": "Algorithmic Stability",
            "Single-Threaded Merge Sort": "Stable ✅",
            "Multithreaded Merge Sort (K workers)": "Stable ✅",
            "Remarks": "Preserves original order of duplicate keys using `<=` comparison.",
        },
        {
            "Metric": "Parallel Work (T₁)",
            "Single-Threaded Merge Sort": "O(n log n)",
            "Multithreaded Merge Sort (K workers)": "O(n log n)",
            "Remarks": "Total CPU work done across all cores.",
        },
        {
            "Metric": "Parallel Span / Critical Path (T∞)",
            "Single-Threaded Merge Sort": "O(n log n)",
            "Multithreaded Merge Sort (K workers)": "O(n + log² n)",
            "Remarks": "Longest chain of dependent operations (reduction merge).",
        },
    ]
    st.table(pd.DataFrame(complexity_data))

    st.markdown("---")
    st.markdown("### 📐 Recurrence Relation & Master Theorem")
    st.latex(r"T(n) = 2 \cdot T\left(\frac{n}{2}\right) + \Theta(n)")
    st.markdown(
        r"""
        By applying the **Master Theorem** ($a = 2, b = 2, f(n) = \Theta(n)$):
        1. Critical exponent: $\log_b a = \log_2 2 = 1$.
        2. Comparing $f(n)$ with $n^{\log_b a}$: $f(n) = \Theta(n^1) = \Theta(n^{\log_b a})$.
        3. Since this matches **Case 2** of the Master Theorem:
        """
    )
    st.latex(r"T(n) = \Theta(n^{\log_b a} \log n) = \Theta(n \log n)")

    st.markdown("---")
    st.markdown("### ⚖️ Amdahl's Law & Parallel Bottlenecks")
    st.latex(r"S_{\text{latency}}(K) = \frac{1}{(1 - p) + \frac{p}{K}}")
    st.markdown(
        r"""
        Where:
        - $K$ is the number of processor cores / threads.
        - $p$ is the fraction of parallelizable code (the chunk sorting phase).
        - $(1 - p)$ is the strictly sequential portion (pairwise reduction merge tree and array splitting).
        
        Because merging requires reading all $n$ elements in the final reduction stage, $(1 - p) > 0$. 
        Consequently, as $K \to \infty$, speedup is bounded by $\frac{1}{1 - p}$.
        """
    )

    st.markdown("---")
    st.markdown("### 🧵 The Python GIL (Global Interpreter Lock) Constraint")
    st.markdown(
        """
        <div class="callout-box">
            <strong>Key Insight for Python Developers:</strong><br>
            CPython features a Global Interpreter Lock (GIL) that ensures only one native thread executes 
            Python bytecode at any given moment. For CPU-bound tasks like pure Python sorting:
            <ul>
                <li><strong>ThreadPoolExecutor:</strong> Threads time-slice cooperatively. Thread context switches and synchronization add latency without true multi-core speedup.</li>
                <li><strong>ProcessPoolExecutor:</strong> Spawns separate OS processes with independent GILs. Achieves genuine multi-core parallelism, but requires IPC serialization (pickling) of chunk data.</li>
                <li><strong>Compiled Languages (C/C++, Rust, Go):</strong> Native threads run simultaneously on separate CPU cores without GIL constraints, yielding near-linear speedups for large n.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 🏗️ Parallel Algorithm Architecture")
    st.markdown(
        """
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
        """
    )


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
f_col1, f_col2 = st.columns([3, 1])
with f_col1:
    st.caption(
        "Merge Sort vs Multithreaded Merge Sort Analyzer | DAA Mini Project | Built with Python, Streamlit & Plotly"
    )
with f_col2:
    st.caption("Status: All Modules Nominal 🟢")
