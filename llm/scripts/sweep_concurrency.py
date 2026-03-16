"""
Sweep over different concurrency levels against two running services
(optimized and unoptimized).

For each service and concurrency value the script runs `vllm bench serve`
and saves results into separate subdirectories.

Usage:
    cd serve-performance-blog/llm/scripts
    python sweep_concurrency.py
"""

import os
import random
import subprocess
import time


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

SERVICES = {
    "optimized": {
        "base_url": "http://localhost:8000",
        "token": "1-RtmzzggZM8tOag-giiHnQjPeg9B9kWANJrB5_UXbQ",
    },
    # "unoptimized": {
    #     "base_url": "http://localhost:8000",
    #     "token": "c1tC6UB28IxL6UP2e8HmSXuxWTHbwgR4VMO8WaHkCSM",
    # },
}

NUM_BATCHES = 16
NUM_WARMUPS = 0
COOLDOWN_S = 15
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

BENCH_CMD_BASE = [
    "vllm",
    "bench",
    "serve",
    "--backend",
    "openai",
    "--model",
    MODEL_ID,
    "--dataset-name",
    "random",
    "--random-input-len",
    "512",
    "--random-output-len",
    "128",
    "--save-result",
    "--append-result",
]

# Resolve paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_DIR = os.path.dirname(SCRIPT_DIR)  # blog/llm/


def run_benchmark(base_url, token, concurrency, seed, result_path):
    """Run vllm bench serve at a given concurrency level."""
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = token

    cmd = BENCH_CMD_BASE + [
        "--base-url",
        base_url,
        "--num-prompts",
        str(concurrency * NUM_BATCHES),
        "--max-concurrency",
        str(concurrency),
        "--num-warmups",
        str(NUM_WARMUPS),
        "--seed",
        str(seed),
        "--result-filename",
        result_path,
    ]

    print(f"Running benchmark: concurrency={concurrency}")
    subprocess.run(cmd, check=True, env=env)


def main():
    for service_name, service_cfg in SERVICES.items():
        results_dir = os.path.join(LLM_DIR, "results", f"concurrency_sweep_{service_name}")
        os.makedirs(results_dir, exist_ok=True)

        print(f"\n{'#' * 60}")
        print(f"  Service: {service_name}")
        print(f"  URL: {service_cfg['base_url']}")
        print(f"{'#' * 60}")

        for concurrency in CONCURRENCY_LEVELS:
            print(f"\n{'=' * 60}")
            print(f"  [{service_name}] Concurrency: {concurrency}")
            print(f"{'=' * 60}\n")

            result_path = os.path.join(results_dir, f"concurrency_{concurrency}.json")
            run_benchmark(
                service_cfg["base_url"],
                service_cfg["token"],
                concurrency,
                seed=random.randint(0, 2**31),
                result_path=result_path,
            )

            if concurrency != CONCURRENCY_LEVELS[-1]:
                print(f"Cooling down for {COOLDOWN_S}s ...")
                time.sleep(COOLDOWN_S)

    print("\nSweep complete.")
    for service_name in SERVICES:
        print(f"  {service_name}: {os.path.join(LLM_DIR, 'results', f'concurrency_sweep_{service_name}')}/")


if __name__ == "__main__":
    main()
