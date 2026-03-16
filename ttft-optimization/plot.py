"""Plot TTFT and TPOT comparison across configurations.

Usage:
    # From result directories (one JSON per concurrency level):
    python plot.py --results-dirs results/vllm_direct results/ray_serve_opt results/ray_serve_default \
        --labels "vLLM (direct)" "Ray Serve w/ Optimizations" "Default Ray Serve"

    # Or with no arguments to use the reference data:
    python plot.py
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]

SCRIPT_DIR = Path(__file__).parent

# Reference data from benchmark runs (ISL=512, OSL=256, TP=4, gemma-3-12b-it, vLLM 0.15.0)
# num_prompts scaled with concurrency: 375 @ c=8, 3000 @ c=64, 12000 @ c=256
CONCURRENCIES = [8, 64, 256]
REFERENCE_DATA = {
    "vLLM (direct)": {
        "ttft": [89.79, 348.30, 447.96],
        "tpot": [6.01, 11.54, 30.54],
    },
    "Ray Serve w/ Optimizations": {
        "ttft": [128.71, 381.52, 448.13],
        "tpot": [5.89, 12.21, 31.80],
    },
    "Default Ray Serve": {
        "ttft": [133.46, 323.06, 1052.77],
        "tpot": [6.09, 11.74, 29.80],
    },
}

COLORS = ["#2E7D32", "#C62828", "#1565C0"]


def load_results(results_dir: str) -> dict:
    """Load per-concurrency JSON files from a results directory."""
    ttft, tpot = [], []
    for c in CONCURRENCIES:
        path = Path(results_dir) / f"c{c}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        data = json.loads(path.read_text())
        ttft.append(data["mean_ttft_ms"])
        tpot.append(data["mean_tpot_ms"])
    return {"ttft": ttft, "tpot": tpot}


def plot(series: dict[str, dict], output: str = "results/ttft_comparison.png"):
    labels = list(series.keys())
    x = np.arange(len(CONCURRENCIES))
    width = 0.24

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("white")

    for ax, metric, ylabel, title in [
        (ax1, "ttft", "Average TTFT (ms)", "TTFT"),
        (ax2, "tpot", "Average TPOT (ms)", "TPOT"),
    ]:
        ax.set_facecolor("white")
        offsets = [(-width, 0), (0, 1), (width, 2)]

        bar_groups = []
        for (offset, ci), label in zip(offsets, labels):
            bars = ax.bar(
                x + offset, series[label][metric], width,
                label=label, color=COLORS[ci],
                edgecolor="white", linewidth=0.8, zorder=3,
            )
            bar_groups.append(bars)

        ax.set_xlabel("Max Concurrency", fontsize=13, fontweight="medium", labelpad=10)
        ax.set_ylabel(ylabel, fontsize=13, fontweight="medium", labelpad=10)
        ax.set_title(f"Average {title} by Concurrency Level", fontsize=14, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(CONCURRENCIES, fontsize=12)
        ax.tick_params(axis="y", labelsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC")
        ax.spines["bottom"].set_color("#CCCCCC")
        ax.yaxis.grid(True, alpha=0.3, linestyle="-", color="#888888", zorder=0)
        ax.xaxis.grid(False)

        for bars in bar_groups:
            for bar in bars:
                h = bar.get_height()
                fmt = f"{h:.0f}" if h >= 10 else f"{h:.1f}"
                ax.annotate(
                    fmt, xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333333",
                )

    handles, legend_labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="upper center", ncol=3, fontsize=11,
        frameon=True, fancybox=True, shadow=False,
        edgecolor="#CCCCCC", facecolor="white", bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle("ISL=512, OSL=256, TP=4, gemma-3-12b-it", fontsize=11, color="#666666", y=1.06)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = SCRIPT_DIR / output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dirs", nargs="*", help="Result directories (one per config)")
    parser.add_argument("--labels", nargs="*", help="Labels for each config")
    parser.add_argument("-o", "--output", default="results/ttft_comparison.png")
    args = parser.parse_args()

    if args.results_dirs:
        if not args.labels or len(args.labels) != len(args.results_dirs):
            parser.error("--labels must match --results-dirs count")
        series = {label: load_results(d) for label, d in zip(args.labels, args.results_dirs)}
    else:
        print("No --results-dirs given, using reference data.")
        series = REFERENCE_DATA

    plot(series, args.output)


if __name__ == "__main__":
    main()
