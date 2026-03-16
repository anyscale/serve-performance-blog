#!/usr/bin/env python3
"""
Visualize RPS sweep results from interactive_rate_bench.

Reads JSON files from results/rps_sweep/ and results/rps_sweep_nonoptimized/
and creates the same 4-panel plot format as visualize_replica_sweep.py.

Usage:
    cd serve-performance-blog/llm/scripts
    python visualize_rps_sweep.py [--qps-per-replica 16]
"""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import re
import sys

QPS_PER_REPLICA = 16

VERSIONS = {
    "nonoptimized": {
        "dir": "rps_sweep_nonoptimized",
        "label": "Default Ray Serve",
        "color": "blue",
        "marker": "o",
    },
    "optimized": {
        "dir": "rps_sweep_optimized",
        "label": "Optimized Ray Serve",
        "color": "green",
        "marker": "D",
    },
}


def load_rps_data(directory, qps_per_replica):
    """Load RPS sweep data and infer replica count from QPS."""
    entries = []

    if not os.path.exists(directory):
        print(f"  Directory {directory} does not exist, skipping")
        return entries

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, "r") as f:
                raw = json.load(f)
            w = raw["window"]

            # Extract target QPS from filename: interactive_measure_qps320p00_...
            qps_match = re.search(r"qps(\d+)p(\d+)", filename)
            if qps_match:
                target_qps = int(qps_match.group(1)) + int(qps_match.group(2)) / 100.0
            else:
                target_qps = w["request_rate"]

            replica_count = round(target_qps / qps_per_replica)
            if replica_count < 1:
                replica_count = 1

            total_output_tokens = int(w["avg_output_tokens"] * w["requests"])
            entries.append({
                "replica_count": replica_count,
                "target_qps": target_qps,
                "actual_qps": w["request_rate"],
                "median_tpot_ms": w["p50_tpot_ms"],
                "median_ttft_ms": w["p50_ttft_ms"],
                "total_output_tokens": total_output_tokens,
                "duration": w["elapsed_s"],
                "num_prompts": w["requests"],
                "throughput_tok_s": w["throughput_tok_s"],
                "model_id": raw.get("config", {}).get("model", "Unknown"),
            })
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: {filepath}: {e}")

    entries.sort(key=lambda x: x["replica_count"])
    return entries


def calculate_metrics(entries):
    """Calculate throughput metrics from entries."""
    replica_counts = [e["replica_count"] for e in entries]
    median_tpot_ms = [e["median_tpot_ms"] for e in entries]
    median_ttft_ms = [e["median_ttft_ms"] for e in entries]
    total_output_throughput = [
        e["total_output_tokens"] / e["duration"] for e in entries
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
        cfg = VERSIONS[version]
        ax.plot(
            m["replica_counts"], m[y_key],
            f"{cfg['marker']}-", linewidth=3, markersize=8,
            color=cfg["color"], label=cfg["label"],
        )
        for x, y in zip(m["replica_counts"], m[y_key]):
            label_text = fmt_fn(y) if fmt_fn else f"{y:.1f}"
            ax.annotate(
                label_text, (x, y),
                textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=8, color=cfg["color"],
            )
    ax.set_xlabel("Number of Replicas")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()


def create_plots(data_by_version, qps_per_replica):
    """Create 4-panel comparison plot."""
    all_replica_counts = set()
    metrics_by_version = {}
    for version, entries in data_by_version.items():
        if entries:
            m = calculate_metrics(entries)
            metrics_by_version[version] = m
            all_replica_counts.update(m["replica_counts"])

    if not all_replica_counts:
        print("Error: No data found")
        return

    replica_counts = sorted(all_replica_counts)

    model_id = "Unknown"
    for entries in data_by_version.values():
        if entries:
            model_id = entries[0]["model_id"]
            break

    replica_range = (
        f"{min(replica_counts)}-{max(replica_counts)}"
        if len(replica_counts) > 1
        else str(replica_counts[0])
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"RPS Sweep: Replica Scaling at {qps_per_replica} QPS/Replica "
        f"({replica_range} Replicas)\nvLLM Backend ({model_id})",
        fontsize=16,
        fontweight="bold",
    )

    def setup_x_axis(ax):
        ax.set_xticks(replica_counts)
        ax.set_xticklabels([str(x) for x in replica_counts])

    _plot_metric(
        axes[0, 0], metrics_by_version,
        "median_tpot_ms", "TPOT (milliseconds)",
        "Time Per Output Token (p50)",
        fmt_fn=lambda y: f"{y:.1f}",
    )
    setup_x_axis(axes[0, 0])

    _plot_metric(
        axes[0, 1], metrics_by_version,
        "median_ttft_ms", "TTFT (milliseconds)",
        "Time To First Token (p50)",
        fmt_fn=lambda y: f"{y:.0f}",
    )
    setup_x_axis(axes[0, 1])

    _plot_metric(
        axes[1, 0], metrics_by_version,
        "total_output_throughput", "Output Tokens/s",
        "Total Output Throughput\n(Total Output Tokens / Duration)",
        fmt_fn=lambda y: f"{y:,.0f}",
    )
    setup_x_axis(axes[1, 0])
    axes[1, 0].yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: f"{int(x):,}")
    )

    _plot_metric(
        axes[1, 1], metrics_by_version,
        "output_per_replica", "Output Tokens/s Per Replica",
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
        "results", "visualizations",
    )
    os.makedirs(vis_dir, exist_ok=True)
    out_path = os.path.join(vis_dir, "rps_sweep_analysis.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")


def print_summary(data_by_version, qps_per_replica):
    """Print text summary."""
    for version, entries in data_by_version.items():
        if not entries:
            continue
        cfg = VERSIONS[version]
        print(f"\n=== {cfg['label']} ({qps_per_replica} QPS/replica) ===\n")
        print(f"{'Replicas':<10} {'Target QPS':<12} {'Actual QPS':<12} {'TPOT p50':<10} {'TTFT p50':<10} {'Throughput':<12}")
        print("-" * 66)
        for e in entries:
            throughput = e["total_output_tokens"] / e["duration"]
            print(
                f"{e['replica_count']:<10} {e['target_qps']:<12.0f} {e['actual_qps']:<12.1f} "
                f"{e['median_tpot_ms']:<10.1f} {e['median_ttft_ms']:<10.0f} {throughput:<12,.0f}"
            )


def main():
    qps_per_replica = QPS_PER_REPLICA

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--qps-per-replica" and i + 1 < len(args):
            qps_per_replica = float(args[i + 1])
            i += 2
        else:
            i += 1

    results_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
    )

    data_by_version = {}
    for version, cfg in VERSIONS.items():
        d = os.path.join(results_base, cfg["dir"])
        print(f"Loading {cfg['label']} from: {d}")
        entries = load_rps_data(d, qps_per_replica)
        data_by_version[version] = entries
        for e in entries:
            print(f"  {e['replica_count']}r @ {e['actual_qps']:.1f} QPS -> {e['throughput_tok_s']:.0f} tok/s")

    has_data = any(data_by_version.values())
    if not has_data:
        print("Error: No data found")
        return

    create_plots(data_by_version, qps_per_replica)
    print_summary(data_by_version, qps_per_replica)


if __name__ == "__main__":
    main()
