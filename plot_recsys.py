#!/usr/bin/env python3
"""
Visualize recsys (DLRM) benchmark results: optimized vs unoptimized nightly.

Usage:
    cd serve-performance-blog
    python plot_recsys.py [--results-dir results/recsys] [--output results_recsys.png]
"""

import argparse
import json
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

LABEL_MAP = {
    "recsys-oss-nightly-unoptimized": "Default Ray Serve",
    "recsys-oss-nightly-optimized": "Ray Serve w/ Optimizations",
}

COLOR_MAP = {
    "recsys-oss-nightly-unoptimized": "#4878CF",
    "recsys-oss-nightly-optimized": "#E45756",
}


def compute_percentile(response_times: dict, p: float) -> float:
    """Compute p-th percentile from a {bucket_ms: count} histogram."""
    total = sum(response_times.values())
    if total == 0:
        return float("nan")
    target = total * p / 100.0
    cumulative = 0
    for bucket in sorted(response_times.keys(), key=float):
        cumulative += response_times[bucket]
        if cumulative >= target:
            return float(bucket)
    return float("nan")


def aggregate_entries(entries: list) -> dict:
    """Aggregate per-URL Locust entries into a single summary."""
    total_requests = sum(e["num_requests"] for e in entries)
    total_failures = sum(e["num_failures"] for e in entries)
    total_response_time = sum(e["total_response_time"] for e in entries)

    if total_requests == 0:
        return None

    # Merge response_times histograms
    merged_rt = Counter()
    for e in entries:
        for bucket, count in e["response_times"].items():
            merged_rt[bucket] += count

    # Compute RPS from per-second counters
    rps_by_sec = Counter()
    for e in entries:
        for ts, count in e["num_reqs_per_sec"].items():
            rps_by_sec[ts] += count

    # Use the overall time window for RPS
    min_start = min(e["start_time"] for e in entries)
    max_end = max(e["last_request_timestamp"] for e in entries)
    duration = max_end - min_start
    rps = total_requests / duration if duration > 0 else 0

    avg_latency = total_response_time / total_requests
    p50 = compute_percentile(merged_rt, 50)
    p95 = compute_percentile(merged_rt, 95)
    p99 = compute_percentile(merged_rt, 99)
    error_rate = total_failures / total_requests * 100

    return {
        "rps": rps,
        "avg_latency": avg_latency,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "total_requests": total_requests,
        "total_failures": total_failures,
        "error_rate": error_rate,
    }


VARIANTS = ["recsys-oss-nightly-optimized", "recsys-oss-nightly-unoptimized"]


def load_results(results_dir: str) -> dict:
    """Load and aggregate all results by variant and concurrency."""
    results = {}
    for variant in VARIANTS:
        variant_dir = os.path.join(results_dir, variant)
        if not os.path.isdir(variant_dir):
            continue
        for fname in sorted(os.listdir(variant_dir)):
            if not fname.endswith(".json"):
                continue
            base = fname.removesuffix(".json")
            if base.endswith(".json"):
                base = base.removesuffix(".json")
            m = re.match(r"^(.+)_(\d+)$", base)
            if not m:
                continue
            concurrency = int(m.group(2))

            path = os.path.join(variant_dir, fname)
            with open(path) as f:
                data = json.load(f)

            if not data:
                continue

            stats = aggregate_entries(data)
            if stats is None:
                continue

            results.setdefault(variant, {})[concurrency] = stats
    return results


def plot(results: dict, output: str):
    """Create a 2x2 publication-quality plot."""
    variants = sorted(results.keys())

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Recommendation System (DLRM) Serving Performance\nDefault Ray Serve vs Ray Serve w/ Optimizations",
        fontsize=15, fontweight="bold", y=0.98,
    )

    metrics = [
        ("rps", "Throughput (RPS)", "Requests / sec"),
        ("avg_latency", "Mean Latency", "Latency (ms)"),
        ("p50", "P50 Latency", "Latency (ms)"),
        ("p99", "P99 Latency", "Latency (ms)"),
    ]

    for ax, (key, title, ylabel) in zip(axes.flat, metrics):
        for variant in variants:
            concurrencies = sorted(results[variant].keys())
            values = [results[variant][c][key] for c in concurrencies]
            label = LABEL_MAP.get(variant, variant)
            color = COLOR_MAP.get(variant, None)
            ax.plot(
                concurrencies, values,
                marker="o", markersize=6, linewidth=2,
                label=label, color=color,
            )

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Concurrent Users")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")

        # Annotate peak/min values
        if key == "rps":
            for variant in variants:
                concurrencies = sorted(results[variant].keys())
                values = [results[variant][c][key] for c in concurrencies]
                max_idx = np.argmax(values)
                ax.annotate(
                    f"{values[max_idx]:.0f}",
                    (concurrencies[max_idx], values[max_idx]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, fontweight="bold",
                    color=COLOR_MAP.get(variant, "black"),
                )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved {output}")

    # Also print a summary table
    print("\n{:<22} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Variant", "Users", "RPS", "Avg(ms)", "P50(ms)", "P95(ms)", "P99(ms)"))
    print("-" * 92)
    for variant in variants:
        label = LABEL_MAP.get(variant, variant)
        for c in sorted(results[variant].keys()):
            s = results[variant][c]
            print("{:<22} {:>8} {:>10.1f} {:>10.1f} {:>10.1f} {:>10.1f} {:>10.1f}".format(
                label, c, s["rps"], s["avg_latency"], s["p50"], s["p95"], s["p99"]))


def parse_args():
    parser = argparse.ArgumentParser(description="Plot recsys benchmark results")
    parser.add_argument(
        "--results-dir", default="results/recsys",
        help="Directory containing variant subdirectories with JSON results",
    )
    parser.add_argument(
        "--output", "-o", default="results_recsys.png",
        help="Output image path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        exit(1)
    plot(results, args.output)
