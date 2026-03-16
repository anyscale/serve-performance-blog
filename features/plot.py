"""
Generate throughput-vs-latency plots for feature performance comparisons.

Reads CSV results produced by sweep.py and generates one PNG per
feature comparison — unary and streaming side-by-side in a 1x2 panel.

Usage:
    python plot.py
    python plot.py --comparison grpc
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# "optimized" variant always comes first, "default" second.
# Colors: optimized = red, default = blue.
COMPARISONS = {
    "grpc": {
        "title": "gRPC",
        "variants": {"on": "gRPC enabled", "off": "gRPC disabled"},
        "roles": {"on": "optimized", "off": "default"},
        "unary_sla_ms": 200,
    },
    "haproxy": {
        "title": "HAProxy",
        "variants": {"on": "HAProxy enabled", "off": "HAProxy disabled"},
        "roles": {"on": "optimized", "off": "default"},
    },
    "gc_eventloop": {
        "title": "Single Event Loop",
        "variants": {
            "optimized": "Optimized (single EL)",
            "unoptimized": "Unoptimized (defaults)",
        },
        "roles": {"optimized": "optimized", "unoptimized": "default"},
        "unary_sla_ms": 125,
    },
    "allon": {
        "title": "All Optimizations",
        "variants": {"on": "All optimizations", "off": "Baseline"},
        "roles": {"on": "optimized", "off": "default"},
    },
    "allon8": {
        "title": "All Optimizations (8 Replicas)",
        "variants": {"on": "All optimizations", "off": "Baseline"},
        "roles": {"on": "optimized", "off": "default"},
    },
}

ROLE_COLORS = {
    "optimized": "#d62728",  # red
    "default": "#1f77b4",    # blue
}

UNARY_SLA_MS = 300
STREAMING_SLA_MS = 700


def load_csv(comparison: str, variant: str, mode: str) -> pd.DataFrame | None:
    path = os.path.join(RESULTS_DIR, f"{comparison}_{variant}_{mode}.csv")
    if not os.path.exists(path):
        print(f"  Skipping {path} (not found)")
        return None
    return pd.read_csv(path)


def plot_throughput_vs_latency(
    ax,
    data_series: list[tuple[str, pd.DataFrame, str, str, str, str]],
    title: str,
    subtitle: str,
    sla_ms: float,
):
    """Plot throughput vs p99 latency with concurrency labels.

    data_series: list of (label, df, throughput_col, latency_col, color, role)
    """
    last_points = {}  # role -> (last_throughput, last_latency)

    for label, df, tp_col, lat_col, color, role in data_series:
        throughput = df[tp_col]
        latency = df[lat_col]
        concurrencies = df["concurrency"]

        # Clip points beyond the SLA
        mask = latency <= sla_ms
        throughput = throughput[mask]
        latency = latency[mask]
        concurrencies = concurrencies[mask]

        ax.plot(latency, throughput, marker="o", label=label, color=color, linewidth=2)

        for x, y, c in zip(latency, throughput, concurrencies):
            ax.annotate(
                str(int(c)),
                (x, y),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=7,
                color=color,
                alpha=0.8,
            )

        if len(throughput) > 0:
            last_points[role] = (throughput.iloc[-1], latency.iloc[-1])

    # Annotate throughput gain between optimized and default
    if "optimized" in last_points and "default" in last_points:
        opt_tp, opt_lat = last_points["optimized"]
        def_tp, def_lat = last_points["default"]
        if def_tp > 0:
            ratio = opt_tp / def_tp
            ratio_str = f"{ratio:.2f}x" if ratio < 1.1 else f"{ratio:.1f}x"
            # Start at the last point with lower latency, arrow down to
            # the other curve's goodput at that same x position
            if opt_lat <= def_lat:
                arrow_x = opt_lat
                arrow_from = opt_tp
                arrow_to = def_tp
            else:
                arrow_x = def_lat
                arrow_from = def_tp
                arrow_to = opt_tp
            ax.annotate(
                "",
                xy=(arrow_x, arrow_to),
                xytext=(arrow_x, arrow_from),
                arrowprops=dict(
                    arrowstyle="<->",
                    color="#333333",
                    lw=1.5,
                ),
            )
            mid_y = (arrow_from + arrow_to) / 2
            ax.text(
                arrow_x,
                mid_y,
                f"{ratio_str} gain  ",
                ha="right",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#333333",
            )

    ax.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)


def generate_combined_plot(comparison: str, cfg: dict):
    # Build unary series
    unary_series = []
    for variant, label in cfg["variants"].items():
        df = load_csv(comparison, variant, "unary")
        if df is not None:
            role = cfg["roles"][variant]
            color = ROLE_COLORS[role]
            unary_series.append((label, df, "rps", "p99_ms", color, role))

    # Build streaming series
    streaming_series = []
    for variant, label in cfg["variants"].items():
        df = load_csv(comparison, variant, "streaming")
        if df is None:
            continue
        df = df.copy()
        df["e2e_p99_ms"] = df["e2e_p99"] * 1000
        role = cfg["roles"][variant]
        color = ROLE_COLORS[role]
        streaming_series.append(
            (label, df, "system_tokens_per_sec", "e2e_p99_ms", color, role)
        )

    if len(unary_series) < 2 and len(streaming_series) < 2:
        print(f"  Not enough data for {comparison}")
        return

    fig, (ax_unary, ax_stream) = plt.subplots(1, 2, figsize=(16, 6))

    unary_sla = cfg.get("unary_sla_ms", UNARY_SLA_MS)
    streaming_sla = cfg.get("streaming_sla_ms", STREAMING_SLA_MS)
    if len(unary_series) >= 2:
        plot_throughput_vs_latency(
            ax_unary,
            unary_series,
            f"{cfg['title']} — Unary (10KB echo)",
            "Plotted by concurrency",
            unary_sla,
        )
        ax_unary.set_xlabel("P99 Latency (ms)")
        ax_unary.set_ylabel(f"Goodput (RPS) @ {unary_sla}ms SLA")
    else:
        ax_unary.set_visible(False)

    if len(streaming_series) >= 2:
        plot_throughput_vs_latency(
            ax_stream,
            streaming_series,
            f"{cfg['title']} — Streaming (10KB in 50 SSE chunks)",
            "Plotted by concurrency",
            streaming_sla,
        )
        ax_stream.set_xlabel("P99 E2E Latency (ms)")
        ax_stream.set_ylabel(f"Goodput (system tokens/sec) @ {streaming_sla}ms SLA")
    else:
        ax_stream.set_visible(False)

    out_path = os.path.join(RESULTS_DIR, f"{comparison}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        choices=sorted(COMPARISONS.keys()),
        default=None,
        help="Which feature to plot (default: all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    comparisons = (
        [args.comparison] if args.comparison else list(COMPARISONS.keys())
    )

    for comp in comparisons:
        cfg = COMPARISONS[comp]
        generate_combined_plot(comp, cfg)


if __name__ == "__main__":
    main()
