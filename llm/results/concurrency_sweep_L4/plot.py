import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Publication style from matplotlib skill
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "axes.linewidth": 1.5,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": 10,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "black",
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_BASE = os.path.dirname(SCRIPT_DIR)  # blog/llm/results/

VARIANTS = {
    "optimized": {
        "dir": os.path.join(RESULTS_BASE, "concurrency_sweep_optimized"),
        "color": "#1f77b4",
        "color2": "#aec7e8",
    },
    "unoptimized": {
        "dir": os.path.join(RESULTS_BASE, "concurrency_sweep_unoptimized"),
        "color": "#d62728",
        "color2": "#ff9896",
    },
}


def load_results(results_dir):
    concurrency = []
    throughput = []
    mean_ttft = []
    mean_tpot = []
    median_ttft = []
    median_tpot = []

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            d = json.load(f)
        concurrency.append(d["max_concurrency"])
        throughput.append(d["output_throughput"])
        mean_ttft.append(d["mean_ttft_ms"])
        mean_tpot.append(d["mean_tpot_ms"])
        median_ttft.append(d["median_ttft_ms"])
        median_tpot.append(d["median_tpot_ms"])

    order = sorted(range(len(concurrency)), key=lambda i: concurrency[i])
    return {
        "concurrency": [concurrency[i] for i in order],
        "throughput": [throughput[i] for i in order],
        "mean_ttft": [mean_ttft[i] for i in order],
        "mean_tpot": [mean_tpot[i] for i in order],
        "median_ttft": [median_ttft[i] for i in order],
        "median_tpot": [median_tpot[i] for i in order],
    }


data = {name: load_results(cfg["dir"]) for name, cfg in VARIANTS.items()}

fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)

# Throughput
ax = axes[0]
for name, cfg in VARIANTS.items():
    d = data[name]
    ax.plot(d["concurrency"], d["throughput"], "o-", color=cfg["color"], label=name)
ax.set_xscale("log", base=2)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
ax.set_xticks(data["optimized"]["concurrency"])
ax.set_xlabel("Max Concurrency")
ax.set_ylabel("Output Tokens/s")
ax.set_title("Throughput vs Concurrency")
ax.legend()

# TTFT
ax = axes[1]
for name, cfg in VARIANTS.items():
    d = data[name]
    ax.plot(d["concurrency"], d["mean_ttft"], "o-", color=cfg["color"], label=f"{name} mean")
    ax.plot(d["concurrency"], d["median_ttft"], "s--", color=cfg["color2"], label=f"{name} p50")
ax.set_xscale("log", base=2)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
ax.set_xticks(data["optimized"]["concurrency"])
ax.set_xlabel("Max Concurrency")
ax.set_ylabel("TTFT (ms)")
ax.set_title("Time to First Token")
ax.legend()

# TPOT
ax = axes[2]
for name, cfg in VARIANTS.items():
    d = data[name]
    ax.plot(d["concurrency"], d["mean_tpot"], "o-", color=cfg["color"], label=f"{name} mean")
    ax.plot(d["concurrency"], d["median_tpot"], "s--", color=cfg["color2"], label=f"{name} p50")
ax.set_xscale("log", base=2)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
ax.set_xticks(data["optimized"]["concurrency"])
ax.set_xlabel("Max Concurrency")
ax.set_ylabel("TPOT (ms)")
ax.set_title("Time per Output Token")
ax.legend()

fig.suptitle(
    "Concurrency Sweep — Qwen2.5-0.5B-Instruct (1 replica)"
    "\ninput=512 tok, output=128 tok",
    fontsize=16,
    fontweight="bold",
)

out_path = os.path.join(SCRIPT_DIR, "concurrency_sweep.png")
plt.savefig(out_path, dpi=300)
print(f"Saved to {out_path}")
