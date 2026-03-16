import json
import os
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
OPTIMIZED_RESULTS_DIR = os.path.join(os.path.dirname(RESULTS_DIR), "replica_sweep_throughput_optimized")

REPLICA_COUNTS = [1, 2, 4, 8, 16, 20]

OPTIMIZED_PATTERN = "{n}_replicas_nightly_optimizations_disabled.json"
DEFAULT_PATTERN = "{n}_replicas_oss_253.json"


def load_throughput(pattern, results_dir=RESULTS_DIR):
    throughputs = []
    for n in REPLICA_COUNTS:
        path = os.path.join(results_dir, pattern.format(n=n))
        with open(path) as f:
            # JSONL: take first entry
            data = json.loads(f.readline())
        throughputs.append(data["output_throughput"])
    return throughputs


optimized = load_throughput(OPTIMIZED_PATTERN, OPTIMIZED_RESULTS_DIR)
default = load_throughput(DEFAULT_PATTERN)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(REPLICA_COUNTS, optimized, "o-", label="Ray Serve w/ Optimizations", color="#D94A4A", linewidth=2, markersize=7)
ax.plot(REPLICA_COUNTS, default, "s-", label="Default Ray Serve", color="#4A90D9", linewidth=2, markersize=7)

ax.set_xlabel("Number of Replicas", fontsize=13)
ax.set_ylabel("Total Output Throughput (tokens/s)", fontsize=13)
ax.set_title("Output Throughput vs. Replica Count\n"
              "ISL=512, OSL=128, Max Concurrency=32/replica",
              fontsize=15)
ax.title.set_linespacing(1.5)
ax.set_xticks(REPLICA_COUNTS)
ax.legend(fontsize=12)
ax.grid(axis="y", alpha=0.3)

# Add value labels on points
for xi, yi in zip(REPLICA_COUNTS, optimized):
    ax.text(xi, yi + 300, f"{yi:.0f}", ha="center", va="bottom", fontsize=8)
for xi, yi in zip(REPLICA_COUNTS, default):
    ax.text(xi, yi + 300, f"{yi:.0f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "replica_sweep_throughput.png"), dpi=150)
plt.show()
