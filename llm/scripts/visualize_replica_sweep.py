#!/usr/bin/env python3
"""
Visualize replica sweep results comparing optimized vs unoptimized nightly.

Adapted from https://github.com/anyscale/custom-router-api-benchmarks

Usage:
    cd serve-performance-blog/llm/scripts
    python visualize_replica_sweep.py [results_dir]
"""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import re
import sys

VERSION_LABELS = {
    "nightly_optimizations_disabled": "Optimizations Disabled",
    "nightly_itl_512_ha_grpc_tpool_tcp_no_delay": "Optimized Ray Serve (Cold Start)",
    "steady_state": "Optimized Ray Serve (Steady State)",
}

VERSION_COLORS = {
    "nightly_optimizations_disabled": "blue",
    "nightly_itl_512_ha_grpc_tpool_tcp_no_delay": "red",
    "steady_state": "green",
}

VERSION_MARKERS = {
    "nightly_optimizations_disabled": "o",
    "nightly_itl_512_ha_grpc_tpool_tcp_no_delay": "s",
    "steady_state": "D",
}


def load_replica_data(directory):
    """Load replica sweep data from directory and organize by version."""
    data = {v: [] for v in VERSION_LABELS}

    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist")
        return data

    for filename in os.listdir(directory):
        # Match steady_state files: <N>_replicas_steady_state.json
        ss_match = re.match(r"(\d+)_replicas_steady_state\.json$", filename)
        if ss_match:
            replica_count = int(ss_match.group(1))
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, "r") as f:
                    raw = json.load(f)
                w = raw["window"]
                # Convert interactive_rate format to vllm bench format
                total_output_tokens = int(
                    w["avg_output_tokens"] * w["requests"]
                )
                entry = {
                    "replica_count": replica_count,
                    "median_tpot_ms": w["p50_tpot_ms"],
                    "median_ttft_ms": w["p50_ttft_ms"],
                    "total_output_tokens": total_output_tokens,
                    "duration": w["elapsed_s"],
                    "num_prompts": w["requests"],
                    "max_concurrency": None,
                    "model_id": raw.get("config", {}).get(
                        "model", "Unknown"
                    ),
                }
                data["steady_state"].append(entry)
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                print(f"  Warning: {filepath}: {e}")
            continue

        for version in VERSION_LABELS:
            if version == "steady_state":
                continue
            match = re.match(
                rf"(\d+)_replicas_{re.escape(version)}\.json$", filename
            )
            if match:
                replica_count = int(match.group(1))
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, "r") as f:
                        content = f.read().strip()
                        # Handle multiple JSON objects (take the last one)
                        if "\n{" in content:
                            json_objects = []
                            for line in content.split("\n"):
                                line = line.strip()
                                if line and line.startswith("{"):
                                    try:
                                        json_objects.append(json.loads(line))
                                    except json.JSONDecodeError:
                                        continue
                            if json_objects:
                                entry = json_objects[-1]
                            else:
                                continue
                        else:
                            entry = json.loads(content)

                        entry["replica_count"] = replica_count
                        data[version].append(entry)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"  Warning: {filepath}: {e}")

    for version in VERSION_LABELS:
        data[version].sort(key=lambda x: x["replica_count"])

    return data


def calculate_metrics(entries):
    """Calculate throughput metrics from benchmark entries."""
    replica_counts = [d["replica_count"] for d in entries]
    median_tpot_ms = [d["median_tpot_ms"] for d in entries]
    median_ttft_ms = [d["median_ttft_ms"] for d in entries]
    num_prompts = [d["num_prompts"] for d in entries]
    max_concurrency = [d["max_concurrency"] for d in entries]

    # Total output throughput from raw duration
    total_output = [d["total_output_tokens"] for d in entries]
    duration = [d["duration"] for d in entries]
    total_output_throughput = [
        out / dur for out, dur in zip(total_output, duration)
    ]

    output_per_replica = [
        t / r for t, r in zip(total_output_throughput, replica_counts)
    ]

    return {
        "replica_counts": replica_counts,
        "median_tpot_ms": median_tpot_ms,
        "median_ttft_ms": median_ttft_ms,
        "total_output_throughput": total_output_throughput,
        "output_per_replica": output_per_replica,
    }


def _plot_metric(ax, metrics_by_version, y_key, ylabel, title, fmt_fn=None):
    """Helper to plot a single metric across versions."""
    for version, m in metrics_by_version.items():
        ax.plot(
            m["replica_counts"],
            m[y_key],
            f"{VERSION_MARKERS[version]}-",
            linewidth=3,
            markersize=8,
            color=VERSION_COLORS[version],
            label=VERSION_LABELS[version],
        )
        for x, y in zip(m["replica_counts"], m[y_key]):
            label_text = fmt_fn(y) if fmt_fn else f"{y:.1f}"
            ax.annotate(
                label_text,
                (x, y),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=8,
                color=VERSION_COLORS[version],
            )
    ax.set_xlabel("Number of Replicas")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()


def create_plots(data):
    """Create comparison plots for optimized vs unoptimized nightly across replica counts."""

    all_replica_counts = set()
    metrics_by_version = {}
    for version in VERSION_LABELS:
        if data[version]:
            m = calculate_metrics(data[version])
            metrics_by_version[version] = m
            all_replica_counts.update(m["replica_counts"])

    if not all_replica_counts:
        print("Error: No data found")
        return

    replica_counts = sorted(all_replica_counts)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    replica_range = (
        f"{min(replica_counts)}-{max(replica_counts)}"
        if len(replica_counts) > 1
        else str(replica_counts[0])
    )

    model_id = "Unknown"
    for version in VERSION_LABELS:
        if data[version]:
            model_id = data[version][0].get("model_id", "Unknown")
            break

    fig.suptitle(
        f"Replica Scaling Comparison ({replica_range} Replicas)\n"
        f"vLLM Backend ({model_id})",
        fontsize=16,
        fontweight="bold",
    )

    def setup_x_axis(ax):
        ax.set_xticks(replica_counts)
        ax.set_xticklabels([str(x) for x in replica_counts])

    # Top Left: TPOT
    _plot_metric(
        axes[0, 0],
        metrics_by_version,
        "median_tpot_ms",
        "TPOT (milliseconds)",
        "Time Per Output Token (p50)",
        fmt_fn=lambda y: f"{y:.1f}",
    )
    setup_x_axis(axes[0, 0])

    # Top Right: TTFT
    _plot_metric(
        axes[0, 1],
        metrics_by_version,
        "median_ttft_ms",
        "TTFT (milliseconds)",
        "Time To First Token (p50)",
        fmt_fn=lambda y: f"{y:.0f}",
    )
    setup_x_axis(axes[0, 1])

    # Bottom Left: Total Output Throughput
    _plot_metric(
        axes[1, 0],
        metrics_by_version,
        "total_output_throughput",
        "Output Tokens/s",
        "Total Output Throughput\n(Total Output Tokens / Duration)",
        fmt_fn=lambda y: f"{y:,.0f}",
    )
    setup_x_axis(axes[1, 0])
    axes[1, 0].yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: f"{int(x):,}")
    )

    # Bottom Right: Throughput Per Replica
    _plot_metric(
        axes[1, 1],
        metrics_by_version,
        "output_per_replica",
        "Output Tokens/s Per Replica",
        "Output Throughput Per Replica",
        fmt_fn=lambda y: f"{y:,.0f}",
    )
    setup_x_axis(axes[1, 1])
    axes[1, 1].yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: f"{int(x):,}")
    )

    plt.tight_layout()

    vis_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        "visualizations",
    )
    os.makedirs(vis_dir, exist_ok=True)
    out_path = os.path.join(vis_dir, "replica_scaling_analysis.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")


def print_summary(data):
    """Print a text summary comparing versions across replica counts."""
    metrics_by_version = {}
    for version in VERSION_LABELS:
        if data[version]:
            metrics_by_version[version] = calculate_metrics(data[version])

    if not metrics_by_version:
        return

    print("\n=== REPLICA SCALING SUMMARY ===\n")

    header = f"{'Replicas':<10}"
    for version in metrics_by_version:
        label = VERSION_LABELS[version]
        header += f" {label + ' TPOT':<16} {label + ' TTFT':<16} {label + ' out tok/s':<18}"
    print(header)
    print("-" * len(header))

    all_rc = set()
    for m in metrics_by_version.values():
        all_rc.update(m["replica_counts"])

    for rc in sorted(all_rc):
        row = f"{rc:<10}"
        for version, m in metrics_by_version.items():
            if rc in m["replica_counts"]:
                idx = m["replica_counts"].index(rc)
                row += f" {m['median_tpot_ms'][idx]:<16.1f} {m['median_ttft_ms'][idx]:<16.0f} {m['total_output_throughput'][idx]:<18.0f}"
            else:
                row += f" {'N/A':<16} {'N/A':<16} {'N/A':<18}"
        print(row)


def main():
    if len(sys.argv) > 1:
        replica_dir = sys.argv[1]
    else:
        replica_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results",
            "replica_sweep_throughput_optimized",
        )

    print(f"Loading data from: {replica_dir}")

    data = load_replica_data(replica_dir)

    has_data = any(data[v] for v in VERSION_LABELS)
    if not has_data:
        print(f"Error: No data found in {replica_dir}/")
        return

    for version in VERSION_LABELS:
        if data[version]:
            rc = [d["replica_count"] for d in data[version]]
            print(f"  {VERSION_LABELS[version]}: replicas {rc}")

    create_plots(data)
    print_summary(data)


if __name__ == "__main__":
    main()
