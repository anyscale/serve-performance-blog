"""
Sweep replica counts for gpt-oss-20b on a single node using ray.serve.llm.

For each replica count:
  1. Deploys locally via serve.run()
  2. Waits for the service to be healthy
  3. Runs vllm bench serve at constant c=256/replica
  4. Shuts down the service

Usage:
    python sweep_replicas.py
"""

import json
import os
import subprocess
import sys
import time

import ray
from ray import serve
from ray.serve.llm import (
    LLMConfig,
    LLMServingArgs,
    ModelLoadingConfig,
    build_openai_app,
)

MODEL_ID = "openai/gpt-oss-20b"
BASE_URL = "http://127.0.0.1:8000"
INPUT_LEN = 512
OUTPUT_LEN = 128
NUM_PROMPTS_PER_REPLICA = 1024
CONCURRENCY_PER_REPLICA = 256
REPLICA_COUNTS = [1, 2, 4, 8]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "replica_sweep_optimized")


def make_app(num_replicas: int):
    """Build a Ray Serve LLM app with the given replica count."""
    llm_config = LLMConfig(
        model_loading_config=ModelLoadingConfig(
            model_id=MODEL_ID,
            model_source=MODEL_ID,
        ),
        engine_kwargs=dict(
            tensor_parallel_size=1,
            max_model_len=4096,
            gpu_memory_utilization=0.95,
            max_num_seqs=256,
            max_num_batched_tokens=16384,
            enable_prefix_caching=False,
        ),
        deployment_config=dict(
            autoscaling_config=dict(
                min_replicas=num_replicas,
                max_replicas=num_replicas,
            ),
            max_ongoing_requests=8192,
        ),
    )

    num_ingress_replicas = num_replicas * 4

    return build_openai_app(
        LLMServingArgs(
            llm_configs=[llm_config],
            ingress_deployment_config={
                "autoscaling_config": {
                    "min_replicas": num_ingress_replicas,
                    "max_replicas": num_ingress_replicas,
                },
            },
        )
    )


def wait_for_healthy(timeout: int = 300):
    """Wait for the serve endpoint to respond to health checks."""
    print("Waiting for service to be healthy...")
    for _ in range(timeout // 2):
        try:
            subprocess.run(
                ["curl", "-sf", f"{BASE_URL}/v1/models"],
                check=True, capture_output=True,
            )
            print("Service is healthy.")
            return
        except subprocess.CalledProcessError:
            time.sleep(2)
    print("Service did not become healthy in time.")
    sys.exit(1)


def run_benchmark(num_replicas: int) -> dict:
    """Run vllm bench serve and return parsed results."""
    result_file = os.path.join(
        RESULTS_DIR, f"{num_replicas}_replicas.json"
    )
    total_concurrency = num_replicas * CONCURRENCY_PER_REPLICA
    total_prompts = num_replicas * NUM_PROMPTS_PER_REPLICA

    cmd = [
        "vllm", "bench", "serve",
        "--backend", "openai",
        "--base-url", BASE_URL,
        "--model", MODEL_ID,
        "--dataset-name", "random",
        "--random-input-len", str(INPUT_LEN),
        "--random-output-len", str(OUTPUT_LEN),
        "--num-prompts", str(total_prompts),
        "--max-concurrency", str(total_concurrency),
        "--request-rate", "inf",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--save-result",
        "--result-filename", result_file,
    ]

    print(f"\n{'='*60}")
    print(f"Benchmarking {num_replicas} replica(s)  "
          f"[c={total_concurrency}, prompts={total_prompts}]")
    print(f"{'='*60}")
    subprocess.run(cmd, check=True)

    with open(result_file) as f:
        return json.load(f)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not ray.is_initialized():
        ray.init()

    summary = []

    for num_replicas in REPLICA_COUNTS:
        print(f"\n{'#'*60}")
        print(f"  Deploying {num_replicas} replica(s)")
        print(f"{'#'*60}\n")

        app = make_app(num_replicas)
        serve.run(app)

        try:
            wait_for_healthy()
            result = run_benchmark(num_replicas)
            row = {
                "replicas": num_replicas,
                "concurrency": num_replicas * CONCURRENCY_PER_REPLICA,
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
        except Exception as e:
            print(f"  FAILED at {num_replicas} replicas: {e}")
            summary.append({"replicas": num_replicas, "error": str(e)})
        finally:
            serve.shutdown()
            time.sleep(10)  # cool-down between deployments

    # Save combined summary
    summary_file = os.path.join(RESULTS_DIR, "sweep_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final table
    print(f"\n{'='*70}")
    print(f"{'Replicas':>10} {'Concurrency':>12} {'RPS':>8} "
          f"{'Out tok/s':>12} {'TTFT(ms)':>10} {'TPOT(ms)':>10}")
    print(f"{'-'*70}")
    for row in summary:
        if "error" in row:
            print(f"{row['replicas']:>10} {'':>12} {'ERROR':>8}")
        else:
            print(f"{row['replicas']:>10} {row['concurrency']:>12} "
                  f"{row['request_throughput']:>8.2f} "
                  f"{row['output_throughput']:>12.1f} "
                  f"{row['mean_ttft_ms']:>10.1f} "
                  f"{row['mean_tpot_ms']:>10.2f}")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
