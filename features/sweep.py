"""
Automatically deploy, benchmark, and tear down services for feature
performance comparisons.

The script:
  1. Deploys ALL required services in parallel
  2. Waits for all of them to reach RUNNING
  3. Runs each concurrency sweep sequentially
  4. Terminates all services

Usage:
    python sweep.py --comparison grpc --mode unary
    python sweep.py --comparison haproxy --mode streaming
    python sweep.py --comparison gc_eventloop --mode both
    python sweep.py                              # run everything
    python sweep.py --resume                     # reuse running services, skip completed benchmarks
"""

import argparse
import csv
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVE_PERF_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # serve-perf/
CONFIGS_DIR = os.path.join(SCRIPT_DIR, "configs")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# ---------------------------------------------------------------------------
# Config-file mapping: (comparison, variant, mode) -> YAML config filename
# ---------------------------------------------------------------------------
CONFIG_MAP = {
    ("grpc", "on", "unary"): "grpc-on-unary.yaml",
    ("grpc", "off", "unary"): "grpc-off-unary.yaml",
    ("grpc", "on", "streaming"): "grpc-on-streaming.yaml",
    ("grpc", "off", "streaming"): "grpc-off-streaming.yaml",
    ("haproxy", "on", "unary"): "haproxy-on-unary.yaml",
    ("haproxy", "off", "unary"): "haproxy-off-unary.yaml",
    ("haproxy", "on", "streaming"): "haproxy-on-streaming.yaml",
    ("haproxy", "off", "streaming"): "haproxy-off-streaming.yaml",
    ("gc_eventloop", "optimized", "unary"): "optimized-unary.yaml",
    ("gc_eventloop", "unoptimized", "unary"): "unoptimized-unary.yaml",
    ("gc_eventloop", "optimized", "streaming"): "optimized-streaming.yaml",
    ("gc_eventloop", "unoptimized", "streaming"): "unoptimized-streaming.yaml",
    ("allon", "on", "unary"): "allon-on-unary.yaml",
    ("allon", "off", "unary"): "allon-off-unary.yaml",
    ("allon", "on", "streaming"): "allon-on-streaming.yaml",
    ("allon", "off", "streaming"): "allon-off-streaming.yaml",
    ("allon8", "on", "unary"): "allon8-on-unary.yaml",
    ("allon8", "off", "unary"): "allon8-off-unary.yaml",
    ("allon8", "on", "streaming"): "allon8-on-streaming.yaml",
    ("allon8", "off", "streaming"): "allon8-off-streaming.yaml",
}

VARIANTS = {
    "grpc": ["on", "off"],
    "haproxy": ["on", "off"],
    "gc_eventloop": ["optimized", "unoptimized"],
    "allon": ["on", "off"],
    "allon8": ["on", "off"],
}

# Comparisons that share an identical "off" baseline service config.
# Only the canonical comparison is deployed+benchmarked; results are copied to others.
SHARED_BASELINE = {
    "canonical": "grpc",
    "members": {"grpc": "off", "haproxy": "off", "allon": "off", "gc_eventloop": "unoptimized"},
}

UNARY_CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
STREAMING_CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64, 128]
LOCUST_PROCESSES = 128
STREAMING_REQUEST_MULTIPLIER = {
    "allon8": 8,
}
STREAMING_MIN_REQUESTS = 3840
STREAMING_NUM_WORKERS = 8
LOCUST_RUN_TIME = 60  # seconds per concurrency level
COOLDOWN_S = 15
SERVICE_WAIT_TIMEOUT = 1800  # 30 min


# ---------------------------------------------------------------------------
# Service lifecycle (deploy / wait / terminate)
# ---------------------------------------------------------------------------
def deploy_service(config_path: str) -> str:
    """Deploy a service and return its name (read from the YAML)."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    service_name = cfg["name"]

    print(f"  Deploying {service_name} from {config_path} ...")
    subprocess.run(
        ["anyscale", "service", "deploy", "-f", config_path],
        check=True,
        cwd=os.path.dirname(config_path),
    )
    return service_name


def wait_for_service(service_name: str) -> tuple[str, str]:
    """Block until RUNNING; return (base_url, token)."""
    print(f"  Waiting for {service_name} ...")
    subprocess.run(
        [
            "anyscale", "service", "wait",
            "--name", service_name,
            "--timeout-s", str(SERVICE_WAIT_TIMEOUT),
        ],
        check=True,
    )

    result = subprocess.run(
        ["anyscale", "service", "status", "--name", service_name, "--json"],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(result.stdout)
    base_url = info["query_url"]
    token = info["query_auth_token"]
    print(f"  RUNNING — {base_url}")
    return base_url, token


def terminate_service(service_name: str):
    """Terminate an Anyscale service."""
    print(f"  Terminating {service_name} ...")
    subprocess.run(
        ["anyscale", "service", "terminate", "--name", service_name],
        check=True,
    )
    print(f"  Terminated {service_name}")


def lookup_running_service(service_name: str) -> tuple[str, str] | None:
    """Check if a service is already RUNNING; return (base_url, token) or None."""
    try:
        result = subprocess.run(
            ["anyscale", "service", "status", "--name", service_name, "--json"],
            capture_output=True, text=True, check=True,
        )
        info = json.loads(result.stdout)
        state = info.get("state", "")
        if state == "RUNNING":
            print(f"  Reusing running service: {service_name}")
            return info["query_url"], info["query_auth_token"]
    except Exception:
        pass
    return None


def deploy_and_wait(config_path: str, resume: bool = False) -> tuple[str, str, str]:
    """Deploy + wait in one call (for use with ThreadPoolExecutor).
    Returns (service_name, base_url, token).
    If resume=True, reuse an already-running service instead of redeploying.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    service_name = cfg["name"]

    if resume:
        existing = lookup_running_service(service_name)
        if existing:
            return service_name, existing[0], existing[1]

    service_name = deploy_service(config_path)
    base_url, token = wait_for_service(service_name)
    return service_name, base_url, token


# ---------------------------------------------------------------------------
# Unary sweep (Locust)
# ---------------------------------------------------------------------------
def run_unary_sweep(comparison: str, variant: str, host: str, token: str):
    """Run Locust-based unary sweep for one variant."""
    label = f"{comparison}_{variant}_unary"
    result_dir = os.path.join(RESULTS_DIR, label)
    os.makedirs(result_dir, exist_ok=True)
    locustfile = os.path.join(SCRIPT_DIR, "locustfile_10kb.py")

    print(f"\n{'#' * 60}")
    print(f"  Unary sweep: {label}")
    print(f"  Host: {host}")
    print(f"{'#' * 60}")

    for concurrency in UNARY_CONCURRENCIES:
        json_file = os.path.join(result_dir, f"{label}_{concurrency}")
        cmd = [
            "locust",
            "--headless",
            "-f", locustfile,
            "-u", str(concurrency),
            "-r", str(min(50, concurrency)),
            "-t", str(LOCUST_RUN_TIME),
            "--processes", str(LOCUST_PROCESSES),
            "--host", host,
            "--json-file", json_file,
            "--reset-stats",
        ]
        if token:
            cmd += [f"--token={token}"]

        print(f"\n  [{label}] concurrency={concurrency}")
        subprocess.run(cmd, check=True)

        if concurrency != UNARY_CONCURRENCIES[-1]:
            print(f"  Cooling down {COOLDOWN_S}s ...")
            time.sleep(COOLDOWN_S)

    _write_unary_csv(result_dir, label)


def _write_unary_csv(result_dir: str, label: str):
    """Parse Locust JSON files and write a summary CSV."""
    csv_path = os.path.join(RESULTS_DIR, f"{label}.csv")
    rows = []

    for concurrency in UNARY_CONCURRENCIES:
        # Locust --json-file appends .json automatically
        json_path = os.path.join(result_dir, f"{label}_{concurrency}.json")
        if not os.path.exists(json_path):
            # Also try without extension (in case of older runs)
            json_path = os.path.join(result_dir, f"{label}_{concurrency}")
        if not os.path.exists(json_path):
            print(f"  Warning: missing {json_path}")
            continue

        with open(json_path) as f:
            raw = f.read()
        # Locust may concatenate multiple JSON values (e.g. "[][{...},{...}]"
        # or "[]  {...}  {...}"). Parse all top-level values and collect dicts.
        decoder = json.JSONDecoder()
        data = []
        idx = 0
        while idx < len(raw):
            raw_stripped = raw[idx:].lstrip()
            if not raw_stripped:
                break
            idx = len(raw) - len(raw_stripped)
            obj, end = decoder.raw_decode(raw, idx)
            if isinstance(obj, list) and obj:
                data = obj
            elif isinstance(obj, dict):
                data.append(obj)
            idx += end

        # Find the "Aggregated" entry
        agg = None
        for entry in data:
            if entry.get("name") == "Aggregated":
                agg = entry
                break
        if agg is None and data:
            agg = data[-1]
        if agg is None:
            continue

        num_requests = agg["num_requests"]
        if num_requests == 0:
            continue

        duration = agg["last_request_timestamp"] - agg["start_time"]
        rps = num_requests / duration if duration > 0 else 0
        mean_latency_ms = agg["total_response_time"] / num_requests
        p50 = _percentile_from_hist(agg["response_times"], num_requests, 50)
        p99 = _percentile_from_hist(agg["response_times"], num_requests, 99)

        rows.append({
            "concurrency": concurrency,
            "rps": round(rps, 2),
            "mean_latency_ms": round(mean_latency_ms, 2),
            "p50_ms": p50,
            "p99_ms": p99,
            "num_requests": num_requests,
        })

    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Wrote {csv_path}")


def _percentile_from_hist(response_times: dict, total: int, pct: float) -> float:
    """Compute a percentile from Locust's response_times histogram."""
    target = total * pct / 100.0
    cumulative = 0
    for bucket in sorted(response_times.keys(), key=lambda k: float(k)):
        cumulative += response_times[bucket]
        if cumulative >= target:
            return float(bucket)
    return 0.0


# ---------------------------------------------------------------------------
# Streaming sweep (llm_stream_benchmark.py)
# ---------------------------------------------------------------------------
def run_streaming_sweep(comparison: str, variant: str, host: str, token: str):
    """Run streaming benchmark for one variant."""
    label = f"{comparison}_{variant}_streaming"
    csv_path = os.path.join(RESULTS_DIR, f"{label}.csv")
    benchmark_script = os.path.join(
        SERVE_PERF_DIR, "streaming", "llm_stream_benchmark.py"
    )
    concurrencies_str = ",".join(str(c) for c in STREAMING_CONCURRENCIES)

    print(f"\n{'#' * 60}")
    print(f"  Streaming sweep: {label}")
    print(f"  Host: {host}")
    print(f"{'#' * 60}")

    request_multiplier = STREAMING_REQUEST_MULTIPLIER.get(comparison, 1)
    max_num_workers = STREAMING_NUM_WORKERS
    min_requests = STREAMING_MIN_REQUESTS
    # Only use multiprocessing for concurrency levels >= num_workers,
    # otherwise workers get concurrency=0 and hang.
    # Split into two runs: low concurrency (single process) and high concurrency (multi-process).
    low_concs = [c for c in STREAMING_CONCURRENCIES if c < max_num_workers]
    high_concs = [c for c in STREAMING_CONCURRENCIES if c >= max_num_workers]

    base_cmd = [
        "python", benchmark_script,
        "--host", host,
        "--path", "/streaming",
        "-tpot", "0",
        "-ttft", "0",
        "-mt", "50",
        "--request-multiplier", str(request_multiplier),
        "--min-requests", str(min_requests),
    ]

    if low_concs:
        low_concs_str = ",".join(str(c) for c in low_concs)
        low_csv = csv_path.replace(".csv", "_low.csv")
        cmd = base_cmd + [
            "--concurrencies", low_concs_str,
            "--output-csv", low_csv,
            "--num-workers", "1",
        ]
        if token:
            cmd += ["--token", token]
        subprocess.run(cmd, check=True)

    if high_concs:
        high_concs_str = ",".join(str(c) for c in high_concs)
        high_csv = csv_path.replace(".csv", "_high.csv") if low_concs else csv_path
        cmd = base_cmd + [
            "--concurrencies", high_concs_str,
            "--output-csv", high_csv,
            "--num-workers", str(max_num_workers),
        ]
        if token:
            cmd += ["--token", token]
        subprocess.run(cmd, check=True)

    # Merge CSVs if we split
    if low_concs and high_concs:
        low_csv = csv_path.replace(".csv", "_low.csv")
        high_csv = csv_path.replace(".csv", "_high.csv")
        import pandas as _pd
        merged = _pd.concat([_pd.read_csv(low_csv), _pd.read_csv(high_csv)])
        merged.to_csv(csv_path, index=False)
        os.remove(low_csv)
        os.remove(high_csv)
    elif low_concs:
        os.rename(csv_path.replace(".csv", "_low.csv"), csv_path)
    # else: high_csv was already written to csv_path

    print(f"  Wrote {csv_path}")


# ---------------------------------------------------------------------------
# Orchestrator: deploy all → benchmark all → terminate all
# ---------------------------------------------------------------------------
def collect_jobs(comparisons: list[str], mode: str) -> list[tuple[str, str, str, str]]:
    """Return list of (comparison, variant, mode, config_path) for all work."""
    jobs = []
    modes = ["unary", "streaming"] if mode == "both" else [mode]
    for comp in comparisons:
        for m in modes:
            for variant in VARIANTS[comp]:
                config_file = CONFIG_MAP[(comp, variant, m)]
                config_path = os.path.join(CONFIGS_DIR, config_file)
                jobs.append((comp, variant, m, config_path))
    return jobs


def _result_csv_exists(comp: str, variant: str, m: str) -> bool:
    """Check if a result CSV already exists for this job."""
    label = f"{comp}_{variant}_{m}"
    csv_path = os.path.join(RESULTS_DIR, f"{label}.csv")
    return os.path.exists(csv_path)


def _is_shared_baseline(comp: str, variant: str) -> bool:
    """Check if this job is a shared baseline (identical config across comparisons)."""
    return comp in SHARED_BASELINE["members"] and SHARED_BASELINE["members"][comp] == variant


def _is_canonical_baseline(comp: str, variant: str) -> bool:
    """Check if this is the canonical job for the shared baseline group."""
    return _is_shared_baseline(comp, variant) and comp == SHARED_BASELINE["canonical"]


def run_all(comparisons: list[str], mode: str, resume: bool = False):
    """Deploy all services in parallel, run benchmarks, tear down."""
    jobs = collect_jobs(comparisons, mode)

    # Separate shared-baseline duplicates from real jobs.
    # Only the canonical baseline is deployed+benchmarked; others get copies.
    canonical_variant = SHARED_BASELINE["members"][SHARED_BASELINE["canonical"]]
    deploy_jobs = []
    copy_jobs = []  # (source_label, target_label)
    for comp, variant, m, config_path in jobs:
        if _is_shared_baseline(comp, variant) and not _is_canonical_baseline(comp, variant):
            source_label = f"{SHARED_BASELINE['canonical']}_{canonical_variant}_{m}"
            target_label = f"{comp}_{variant}_{m}"
            copy_jobs.append((source_label, target_label))
        else:
            deploy_jobs.append((comp, variant, m, config_path))

    if resume:
        skipped = [(c, v, m) for c, v, m, _ in deploy_jobs if _result_csv_exists(c, v, m)]
        if skipped:
            print(f"\n  Skipping {len(skipped)} completed benchmarks:")
            for c, v, m in skipped:
                print(f"    - {c}_{v}_{m}")
        deploy_jobs = [
            j for j in deploy_jobs if not _result_csv_exists(j[0], j[1], j[2])
        ]

    if not deploy_jobs:
        print("  All benchmarks already completed!")
        return

    print(f"\n{'=' * 60}")
    print(f"  Deploying/locating {len(deploy_jobs)} services ...")
    print(f"{'=' * 60}")

    # Phase 1: Deploy + wait for all services in parallel
    ready = {}
    with ThreadPoolExecutor(max_workers=len(deploy_jobs)) as pool:
        futures = {
            pool.submit(deploy_and_wait, config_path, resume): (comp, variant, m, config_path)
            for comp, variant, m, config_path in deploy_jobs
        }
        for future in as_completed(futures):
            comp, variant, m, config_path = futures[future]
            try:
                service_name, base_url, token = future.result()
                ready[config_path] = (service_name, base_url, token)
            except Exception as e:
                print(f"  FAILED to deploy {comp}/{variant}/{m}: {e}")

    print(f"\n{'=' * 60}")
    print(f"  {len(ready)}/{len(deploy_jobs)} services ready. Running benchmarks ...")
    print(f"{'=' * 60}")

    # Phase 2: Run benchmarks sequentially (one at a time for clean results)
    try:
        for comp, variant, m, config_path in deploy_jobs:
            if config_path not in ready:
                print(f"  Skipping {comp}/{variant}/{m} (deploy failed)")
                continue

            _, base_url, token = ready[config_path]

            if m == "unary":
                run_unary_sweep(comp, variant, base_url, token)
            else:
                run_streaming_sweep(comp, variant, base_url, token)

    finally:
        # Copy shared baseline results to other comparisons (always runs)
        import shutil
        for source_label, target_label in copy_jobs:
            src = os.path.join(RESULTS_DIR, f"{source_label}.csv")
            dst = os.path.join(RESULTS_DIR, f"{target_label}.csv")
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"  Copied {source_label}.csv -> {target_label}.csv")
            else:
                print(f"  Warning: cannot copy {source_label}.csv (not found)")
        # Skip service termination for debugging
        print(f"\n  Services left running for debugging.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--comparison",
        choices=["grpc", "haproxy", "gc_eventloop", "allon", "allon8"],
        default=None,
        help="Which feature comparison to run (default: all)",
    )
    parser.add_argument(
        "--mode",
        choices=["unary", "streaming", "both"],
        default="both",
        help="Benchmark mode (default: both)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse already-running services and skip completed benchmarks",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    comparisons = [args.comparison] if args.comparison else list(VARIANTS.keys())
    run_all(comparisons, args.mode, resume=args.resume)

    print("\nAll sweeps complete. Results in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
