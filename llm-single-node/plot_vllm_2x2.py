"""2x2 plot of TTFT, TPOT, derived input tok/s, derived output tok/s vs concurrency."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_vllm")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "vllm_2x2.png")

INPUT_LEN = 512
OUTPUT_LEN = 128


def load_results():
    """Load all concurrency_*.json files, return sorted lists."""
    concurrencies = []
    data = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.startswith("concurrency_") and fname.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, fname)) as f:
                d = json.load(f)
            concurrencies.append(d["max_concurrency"])
            data.append(d)
    order = np.argsort(concurrencies)
    return [concurrencies[i] for i in order], [data[i] for i in order]


def main():
    concurrencies, data = load_results()

    ttft = [d["mean_ttft_ms"] for d in data]
    tpot = [d["mean_tpot_ms"] for d in data]

    # Derived: input tok/s = (input_len / mean_ttft_s) * concurrency
    input_toks = [(INPUT_LEN / (d["mean_ttft_ms"] / 1000)) * c
                  for c, d in zip(concurrencies, data)]

    # Derived: output tok/s = (1 / mean_tpot_s) * concurrency
    output_toks = [(1.0 / (d["mean_tpot_ms"] / 1000)) * c
                   for c, d in zip(concurrencies, data)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "vLLM Single GPU Performance vs Concurrency\n"
        "openai/gpt-oss-20b | ISL=512, OSL=128 | 1xH100",
        fontsize=14,
    )

    # TTFT
    ax = axes[0, 0]
    ax.plot(concurrencies, ttft, "o-", color="#E53935", linewidth=2, markersize=7)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Mean TTFT (ms)")
    ax.set_title("Time to First Token")
    ax.grid(True, alpha=0.3)
    for c, v in zip(concurrencies, ttft):
        ax.annotate(f"{v:.0f}", (c, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    # TPOT
    ax = axes[0, 1]
    ax.plot(concurrencies, tpot, "s-", color="#2196F3", linewidth=2, markersize=7)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Mean TPOT (ms)")
    ax.set_title("Time per Output Token")
    ax.grid(True, alpha=0.3)
    for c, v in zip(concurrencies, tpot):
        ax.annotate(f"{v:.1f}", (c, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    # Derived input tok/s
    ax = axes[1, 0]
    ax.plot(concurrencies, input_toks, "D-", color="#4CAF50", linewidth=2, markersize=7)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Input tok/s")
    ax.set_title("Derived Input Throughput\n(ISL / mean_TTFT_s) * concurrency")
    ax.grid(True, alpha=0.3)
    for c, v in zip(concurrencies, input_toks):
        ax.annotate(f"{v:,.0f}", (c, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    # Derived output tok/s
    ax = axes[1, 1]
    ax.plot(concurrencies, output_toks, "^-", color="#FF9800", linewidth=2, markersize=7)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Output tok/s")
    ax.set_title("Derived Output Throughput\n(1 / mean_TPOT_s) * concurrency")
    ax.grid(True, alpha=0.3)
    for c, v in zip(concurrencies, output_toks):
        ax.annotate(f"{v:,.0f}", (c, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    # Add headroom for annotations
    for ax in axes.flat:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(bottom=0, top=ymax * 1.15)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved plot to {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
