"""Plot output throughput vs replica count for optimized, unoptimized, and vLLM baseline.

Usage:
    python plot_throughput.py              # without vLLM baseline
    python plot_throughput.py --vllm       # with vLLM baseline
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPTIMIZED_DIR = os.path.join(SCRIPT_DIR, "results", "replica_sweep_optimized")
UNOPTIMIZED_DIR = os.path.join(SCRIPT_DIR, "results", "replica_sweep_unoptimized")
VLLM_RESULT = os.path.join(SCRIPT_DIR, "results_vllm", "concurrency_256.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "throughput_vs_replicas.png")


def load_sweep(results_dir):
    """Load replica sweep results, return sorted (replicas, output_throughput) lists."""
    replicas = []
    throughputs = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith("_replicas.json"):
            n = int(fname.split("_")[0])
            with open(os.path.join(results_dir, fname)) as f:
                data = json.load(f)
            replicas.append(n)
            throughputs.append(data["output_throughput"])
    order = np.argsort(replicas)
    return np.array(replicas)[order], np.array(throughputs)[order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", action="store_true",
                        help="Include scaled vLLM baseline line")
    args = parser.parse_args()

    # Load data
    opt_replicas, opt_throughput = load_sweep(OPTIMIZED_DIR)
    unopt_replicas, unopt_throughput = load_sweep(UNOPTIMIZED_DIR)

    all_replicas = sorted(set(list(opt_replicas) + list(unopt_replicas)))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(opt_replicas, opt_throughput, "o-", linewidth=2, markersize=8,
            color="#E53935", label="Ray Serve w/ Optimizations")
    ax.plot(unopt_replicas, unopt_throughput, "s-", linewidth=2, markersize=8,
            color="#2196F3", label="Default Ray Serve")

    # Optional vLLM baseline
    if args.vllm and os.path.exists(VLLM_RESULT):
        with open(VLLM_RESULT) as f:
            vllm_data = json.load(f)
        vllm_single_throughput = vllm_data["output_throughput"]
        vllm_scaled = [vllm_single_throughput * r for r in all_replicas]
        ax.plot(all_replicas, vllm_scaled, "d--", linewidth=2, markersize=8,
                color="#4CAF50", label=f"vLLM baseline (scaled, {vllm_single_throughput:.0f} tok/s/replica)")

    ax.set_xlabel("Replica Count", fontsize=13)
    ax.set_ylabel("Output Throughput (tok/s)", fontsize=13)
    ax.set_title(
        "Output Throughput vs Replica Count\n"
        "openai/gpt-oss-20b | ISL=512, OSL=128 | c=256/replica | 1xH100/replica",
        fontsize=13,
    )
    ax.set_xticks(all_replicas)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    # Add headroom above the highest data point for annotations
    all_values = list(opt_throughput) + list(unopt_throughput)
    if args.vllm and os.path.exists(VLLM_RESULT):
        all_values += vllm_scaled
    ax.set_ylim(bottom=0, top=max(all_values) * 1.15)

    # Annotate data points
    for r, t in zip(opt_replicas, opt_throughput):
        ax.annotate(f"{t:,.0f}", (r, t), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9, color="#E53935")
    for r, t in zip(unopt_replicas, unopt_throughput):
        ax.annotate(f"{t:,.0f}", (r, t), textcoords="offset points",
                    xytext=(0, -18), ha="center", fontsize=9, color="#2196F3")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved plot to {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
