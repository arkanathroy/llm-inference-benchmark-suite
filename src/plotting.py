"""
plotting.py
===========
Chart generation for the benchmark report. Uses Plotly for interactive
notebook display and static PNG export for the README/report.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


PALETTE = {
    "fp16": "#4f98a3",
    "int8": "#d19900",
    "gptq": "#a12c7b",
    "awq": "#437a22",
}


def plot_ttft_by_concurrency(df: pd.DataFrame, out_path: str):
    """
    Line chart: TTFT P95 (y) vs concurrency level (x), one line per
    precision method. This is the primary chart for identifying the
    "performance cliff" -- the concurrency level where TTFT P95
    inflects sharply upward marks the point past which
    max_num_seqs/KV-cache capacity is saturated and requests start
    queueing rather than executing immediately.
    """
    fig = go.Figure()
    for precision in df["precision"].unique():
        sub = df[df["precision"] == precision].sort_values("concurrency")
        fig.add_trace(go.Scatter(
            x=sub["concurrency"], y=sub["ttft_p95_ms"],
            mode="lines+markers", name=precision,
            line=dict(color=PALETTE.get(precision, "#888")),
        ))
    fig.update_layout(
        title="TTFT P95 vs Concurrency (Performance Cliff Detection)",
        xaxis_title="Concurrent Requests", yaxis_title="TTFT P95 (ms)",
        xaxis_type="log", template="plotly_white",
    )
    fig.write_image(out_path, width=900, height=550, scale=2)
    return fig


def plot_throughput_by_concurrency(df: pd.DataFrame, out_path: str):
    fig = go.Figure()
    for precision in df["precision"].unique():
        sub = df[df["precision"] == precision].sort_values("concurrency")
        fig.add_trace(go.Scatter(
            x=sub["concurrency"], y=sub["aggregate_throughput_tok_s"],
            mode="lines+markers", name=precision,
            line=dict(color=PALETTE.get(precision, "#888")),
        ))
    fig.update_layout(
        title="Aggregate Throughput vs Concurrency",
        xaxis_title="Concurrent Requests", yaxis_title="Tokens/sec",
        xaxis_type="log", template="plotly_white",
    )
    fig.write_image(out_path, width=900, height=550, scale=2)
    return fig


def plot_accuracy_vs_memory(df: pd.DataFrame, out_path: str):
    """
    Scatter: memory footprint (x) vs MMLU accuracy (y), sized by
    throughput -- a single chart visualizing the full 3-way tradeoff
    (memory, accuracy, speed). The ideal quantization method sits in
    the top-left (low memory, high accuracy retained) with a large
    marker (high throughput).
    """
    fig = px.scatter(
        df, x="memory_mb", y="mmlu_accuracy", size="throughput_tok_s",
        color="precision", color_discrete_map=PALETTE,
        text="precision", template="plotly_white",
        title="Quantization Tradeoff: Memory vs Accuracy vs Throughput (marker size)",
    )
    fig.update_traces(textposition="top center")
    fig.write_image(out_path, width=900, height=600, scale=2)
    return fig


def plot_cost_per_million_tokens(df: pd.DataFrame, out_path: str):
    fig = px.bar(
        df, x="instance", y="cost_per_million_tokens", color="precision",
        color_discrete_map=PALETTE, barmode="group", template="plotly_white",
        title="Cost per Million Tokens by Instance and Precision",
    )
    fig.update_layout(yaxis_title="USD per 1M tokens", xaxis_title="Instance Type")
    fig.write_image(out_path, width=1000, height=550, scale=2)
    return fig


def save_all_results(results: dict, out_dir: str = "benchmarks"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    import json
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
