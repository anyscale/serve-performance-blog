#!/usr/bin/env python3
"""
Bar chart of peak performance gains: Ray Serve w/ Optimizations vs Default.

Left group:  throughput speed-up (x) for RecSys, LLM, and gRPC streaming.
Right group: latency as fraction of original for RecSys, LLM, and gRPC streaming.

Usage:
    cd serve-perf/blog
    python plot_peak_gains.py
"""

import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLOG_DIR = os.path.dirname(os.path.abspath(__file__))

# ── RecSys ────────────────────────────────────────────────────────────────────

def _recsys_stats(variant, concurrency):
    """Return (rps, p99_latency_ms) for a given variant + concurrency."""
    vdir = os.path.join(BLOG_DIR, "results", "recsys", variant)
    for fname in os.listdir(vdir):
        if not fname.endswith(".json"):
            continue
        m = re.match(r"^(.+)_(\d+)$", fname.removesuffix(".json"))
        if not m or int(m.group(2)) != concurrency:
            continue
        with open(os.path.join(vdir, fname)) as f:
            data = json.load(f)
        if not data:
            continue
        tr = sum(e["num_requests"] for e in data)
        ms = min(e["start_time"] for e in data)
        me = max(e["last_request_timestamp"] for e in data)
        dur = me - ms
        rps = tr / dur if dur > 0 else 0
        # Compute p99 from response_times histograms
        all_times = []
        for entry in data:
            for ms_str, count in entry["response_times"].items():
                all_times.extend([float(ms_str)] * count)
        all_times.sort()
        p99 = np.percentile(all_times, 99) if all_times else 0
        return rps, p99
    return 0, 0


RECSYS_CONC = 400
recsys_def_rps, recsys_def_lat = _recsys_stats("recsys-oss-nightly-unoptimized", RECSYS_CONC)
recsys_opt_rps, recsys_opt_lat = _recsys_stats("recsys-oss-nightly-optimized", RECSYS_CONC)

# ── LLM ───────────────────────────────────────────────────────────────────────

def _llm_stats(pattern, results_dir, n):
    path = os.path.join(results_dir, pattern.format(n=n))
    with open(path) as f:
        d = json.loads(f.readline())
    return d["output_throughput"], d["p99_ttft_ms"]


LLM_REPLICAS = 16
llm_def_tp, llm_def_ttft = _llm_stats(
    "{n}_replicas_oss_253.json",
    os.path.join(BLOG_DIR, "llm", "results", "replica_sweep"),
    LLM_REPLICAS,
)
llm_opt_tp, llm_opt_ttft = _llm_stats(
    "{n}_replicas_nightly_optimizations_disabled.json",
    os.path.join(BLOG_DIR, "llm", "results", "replica_sweep_throughput_optimized"),
    LLM_REPLICAS,
)

# ── gRPC Streaming ────────────────────────────────────────────────────────────

GRPC_SLA_MS = 700  # same SLA used in features/plot.py

def _grpc_streaming_peak(variant):
    """Return (peak_system_tokens_per_sec, e2e_p99_ms_at_peak) within SLA."""
    path = os.path.join(BLOG_DIR, "features", "results", f"grpc_{variant}_streaming.csv")
    df = pd.read_csv(path)
    df["e2e_p99_ms"] = df["e2e_p99"] * 1000
    within_sla = df[df["e2e_p99_ms"] <= GRPC_SLA_MS]
    peak_idx = within_sla["system_tokens_per_sec"].idxmax()
    return within_sla.loc[peak_idx, "system_tokens_per_sec"], within_sla.loc[peak_idx, "e2e_p99_ms"]


grpc_def_tp, grpc_def_lat = _grpc_streaming_peak("off")
grpc_opt_tp, grpc_opt_lat = _grpc_streaming_peak("on")

# ── All Optimizations (streaming) ─────────────────────────────────────────────

ALLON_SLA_MS = 700  # same SLA used in features/plot.py

def _allon_streaming_load(variant):
    """Return DataFrame with e2e_p99_ms column added."""
    path = os.path.join(BLOG_DIR, "features", "results", f"allon8_{variant}_streaming.csv")
    df = pd.read_csv(path)
    df["e2e_p99_ms"] = df["e2e_p99"] * 1000
    return df


def _allon_streaming_peak(df):
    """Return (peak_tp, concurrency_at_peak, e2e_p99_at_peak) within SLA."""
    within_sla = df[df["e2e_p99_ms"] <= ALLON_SLA_MS]
    peak_idx = within_sla["system_tokens_per_sec"].idxmax()
    row = within_sla.loc[peak_idx]
    return row["system_tokens_per_sec"], int(row["concurrency"]), row["e2e_p99_ms"]


allon_def_df = _allon_streaming_load("off")
allon_opt_df = _allon_streaming_load("on")
allon_def_tp, allon_lat_conc, _ = _allon_streaming_peak(allon_def_df)
allon_opt_tp, _, _ = _allon_streaming_peak(allon_opt_df)
allon_def_lat = allon_def_df[allon_def_df["concurrency"] == allon_lat_conc]["e2e_p99_ms"].iloc[0]
allon_opt_lat = allon_opt_df[allon_opt_df["concurrency"] == allon_lat_conc]["e2e_p99_ms"].iloc[0]

# ── All Optimizations (unary) ────────────────────────────────────────────────

def _allon_unary_load(variant):
    """Return DataFrame for unary CSV."""
    path = os.path.join(BLOG_DIR, "features", "results", f"allon8_{variant}_unary.csv")
    return pd.read_csv(path)


def _allon_unary_peak(df):
    """Return (peak_rps, concurrency_at_peak)."""
    peak_idx = df["rps"].idxmax()
    row = df.loc[peak_idx]
    return row["rps"], int(row["concurrency"])


allon_u_def_df = _allon_unary_load("off")
allon_u_opt_df = _allon_unary_load("on")
allon_u_def_rps, allon_u_lat_conc = _allon_unary_peak(allon_u_def_df)
allon_u_opt_rps, _ = _allon_unary_peak(allon_u_opt_df)
allon_u_def_lat = allon_u_def_df[allon_u_def_df["concurrency"] == allon_u_lat_conc]["p99_ms"].iloc[0]
allon_u_opt_lat = allon_u_opt_df[allon_u_opt_df["concurrency"] == allon_u_lat_conc]["p99_ms"].iloc[0]

# ── Derived metrics ───────────────────────────────────────────────────────────

recsys_speedup = recsys_opt_rps / recsys_def_rps
llm_speedup = llm_opt_tp / llm_def_tp
grpc_speedup = grpc_opt_tp / grpc_def_tp
allon_speedup = allon_opt_tp / allon_def_tp
allon_u_speedup = allon_u_opt_rps / allon_u_def_rps

print(f"RecSys @{RECSYS_CONC} users – RPS: {recsys_def_rps:.0f} → {recsys_opt_rps:.0f}  ({recsys_speedup:.1f}x)")
print(f"  P99 latency: {recsys_def_lat:.0f} → {recsys_opt_lat:.0f} ms")
print(f"LLM @{LLM_REPLICAS} replicas – Throughput: {llm_def_tp:.0f} → {llm_opt_tp:.0f} tok/s  ({llm_speedup:.1f}x)")
print(f"  P99 TTFT: {llm_def_ttft:.0f} → {llm_opt_ttft:.0f} ms")
print(f"gRPC Streaming – Peak goodput: {grpc_def_tp:.0f} → {grpc_opt_tp:.0f} tok/s  ({grpc_speedup:.1f}x)")
print(f"  E2E P99: {grpc_def_lat:.0f} → {grpc_opt_lat:.0f} ms")
print(f"All Opts Streaming – Peak goodput: {allon_def_tp:.0f} → {allon_opt_tp:.0f} tok/s  ({allon_speedup:.1f}x)")
print(f"  E2E P99 @conc={allon_lat_conc}: {allon_def_lat:.0f} → {allon_opt_lat:.0f} ms")
print(f"All Opts Unary – Peak RPS: {allon_u_def_rps:.0f} → {allon_u_opt_rps:.0f}  ({allon_u_speedup:.1f}x)")
print(f"  P99 latency @conc={allon_u_lat_conc}: {allon_u_def_lat:.0f} → {allon_u_opt_lat:.0f} ms")

# ── Plot ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle(
    "Peak Performance Gains: Optimizations Enabled vs. Defaults",
    fontsize=15, fontweight="bold", y=0.98,
)
fig.text(
    0.5, 0.91,
    "RAY_SERVE_ENABLE_HA_PROXY=1   RAY_SERVE_THROUGHPUT_OPTIMIZED=1",
    ha="center", fontsize=11, fontstyle="italic", family="monospace", color="#555",
)
fig.subplots_adjust(top=0.85)

import seaborn as sns
colors = sns.color_palette("crest", 4)
x = np.array([0, 1, 2, 3])
width = 0.5
labels = ["RecSys", "LLM", "No-Op Unary", "Echo Streaming"]

# ── Left: Throughput speed-up ─────────────────────────────────────────────────
speedups = [recsys_speedup, llm_speedup, allon_u_speedup, allon_speedup]
bars = ax1.bar(x, speedups, width, color=colors)
ax1.set_ylabel("Throughput Speed-Up", fontsize=13)
ax1.set_title("Throughput Gain (higher is better)", fontsize=13, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylim(0, max(speedups) * 1.3)
ax1.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
ax1.grid(axis="y", alpha=0.3, linestyle="--")
for bar, val in zip(bars, speedups):
    ax1.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
        f"{val:.1f}x", ha="center", fontsize=14, fontweight="bold",
    )

# ── Right: P99 latency as fraction of original (lower is better) ──────────────
lat_fracs = [
    recsys_opt_lat / recsys_def_lat,
    llm_opt_ttft / llm_def_ttft,
    allon_u_opt_lat / allon_u_def_lat,
    allon_opt_lat / allon_def_lat,
]
bars = ax2.bar(x, lat_fracs, width, color=colors)
ax2.set_ylabel("P99 Latency (fraction of baseline)", fontsize=13)
ax2.set_title("P99 Latency Reduction (lower is better)", fontsize=13, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_ylim(0, 1.15)
ax2.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
ax2.grid(axis="y", alpha=0.3, linestyle="--")
for bar, val in zip(bars, lat_fracs):
    ax2.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
        f"{val:.2f}x", ha="center", fontsize=14, fontweight="bold",
    )

plt.tight_layout(rect=[0, 0, 1, 0.92])
out = os.path.join(BLOG_DIR, "peak_performance_gains.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved {out}")
