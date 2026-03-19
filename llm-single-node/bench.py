"""Sweep max concurrency to find peak RPS for gpt-oss-20b on a single H100."""

import json
import subprocess
import sys
import time

MODEL = "openai/gpt-oss-20b"
BASE_URL = "http://127.0.0.1:8000"
INPUT_LEN = 512
OUTPUT_LEN = 128
NUM_PROMPTS = 1024
CONCURRENCY_LEVELS = [16, 32, 64, 128, 256]
RESULTS_DIR = "results"


def run_bench(concurrency: int) -> dict:
    """Run vllm bench serve at a given concurrency and return parsed results."""
    result_file = f"{RESULTS_DIR}/concurrency_{concurrency}.json"
    cmd = [
        "vllm", "bench", "serve",
        "--backend", "vllm",
        "--base-url", BASE_URL,
        "--model", MODEL,
        "--dataset-name", "random",
        "--random-input-len", str(INPUT_LEN),
        "--random-output-len", str(OUTPUT_LEN),
        "--num-prompts", str(NUM_PROMPTS),
        "--max-concurrency", str(concurrency),
        "--request-rate", "inf",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--save-result",
        "--result-filename", result_file,
    ]
    print(f"\n{'='*60}")
    print(f"Running concurrency={concurrency}")
    print(f"{'='*60}")
    subprocess.run(cmd, check=True)

    with open(result_file) as f:
        return json.load(f)


def main():
    subprocess.run(["mkdir", "-p", RESULTS_DIR], check=True)

    # Wait for server to be ready
    print("Waiting for vLLM server to be ready...")
    for _ in range(60):
        try:
            subprocess.run(
                ["curl", "-sf", f"{BASE_URL}/health"],
                check=True, capture_output=True,
            )
            print("Server is ready.")
            break
        except subprocess.CalledProcessError:
            time.sleep(2)
    else:
        print("Server not ready after 120s, exiting.")
        sys.exit(1)

    summary = []
    for concurrency in CONCURRENCY_LEVELS:
        try:
            result = run_bench(concurrency)
            row = {
                "concurrency": concurrency,
                "request_throughput": result.get("request_throughput"),
                "output_throughput": result.get("output_throughput"),
                "total_throughput": result.get("total_throughput"),
                "mean_ttft_ms": result.get("mean_ttft_ms"),
                "mean_tpot_ms": result.get("mean_tpot_ms"),
                "mean_e2el_ms": result.get("mean_e2el_ms"),
            }
            summary.append(row)
            print(f"  RPS: {row['request_throughput']:.2f}  |  "
                  f"Output tok/s: {row['output_throughput']:.1f}  |  "
                  f"TTFT: {row['mean_ttft_ms']:.1f}ms  |  "
                  f"TPOT: {row['mean_tpot_ms']:.2f}ms")
        except (subprocess.CalledProcessError, KeyError, json.JSONDecodeError) as e:
            print(f"  FAILED at concurrency={concurrency}: {e}")
            summary.append({"concurrency": concurrency, "error": str(e)})

    # Save combined summary
    summary_file = f"{RESULTS_DIR}/sweep_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final table
    print(f"\n{'='*60}")
    print(f"{'Concurrency':>12} {'RPS':>8} {'Out tok/s':>12} {'TTFT(ms)':>10} {'TPOT(ms)':>10}")
    print(f"{'-'*60}")
    for row in summary:
        if "error" in row:
            print(f"{row['concurrency']:>12} {'ERROR':>8}")
        else:
            print(f"{row['concurrency']:>12} {row['request_throughput']:>8.2f} "
                  f"{row['output_throughput']:>12.1f} "
                  f"{row['mean_ttft_ms']:>10.1f} {row['mean_tpot_ms']:>10.2f}")

    print(f"\nResults saved to {summary_file}")


if __name__ == "__main__":
    main()
