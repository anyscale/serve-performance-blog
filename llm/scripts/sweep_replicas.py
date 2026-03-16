"""
Sweep over different replica counts, deploying Anyscale services with
pre-built container images.

For each (version, replica_count) combination the script:
  1. Generates a service YAML config (image_uri + inlined compute)
  2. Deploys it via `anyscale service deploy`
  3. Waits for RUNNING status
  4. Runs `vllm bench serve` against the service endpoint
  5. Terminates the service

Usage:
    cd serve-performance-blog/llm/scripts
    python sweep_replicas.py
"""

import json
import os
import subprocess
import time

import yaml


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Version configs: each entry specifies an "image_uri" for a pre-built
# container image.
VERSIONS = {
    "nightly_optimizations_disabled": {
        "image_uri": "anyscale/ray-llm:nightly-py311-cu128",
    },
}

# EXPERIMENT_A_ENV_VARS = {
#     "RAY_SERVE_ENABLE_HA_PROXY": "0",
#     "RAY_SERVE_HAPROXY_TCP_NODELAY": "0",
#     "RAY_SERVE_RUN_SYNC_IN_THREADPOOL": "0",
#     "RAY_SERVE_USE_GRPC_BY_DEFAULT": "0",
# }
EXPERIMENT_B_ENV_VARS = {
    "RAY_SERVE_ENABLE_HA_PROXY": "1",
    "RAY_SERVE_HAPROXY_TCP_NODELAY": "1",
    "RAY_SERVE_THROUGHPUT_OPTIMIZED": "1",
}

# Inline compute config (avoids needing a pre-registered named config).
COMPUTE_CONFIG = {
    "cloud": "anyscale_v2_default_cloud",
    "head_node": {"instance_type": "m8a.xlarge"},
    "worker_nodes": [
        {
            "instance_type": "g6.12xlarge",
            "min_nodes": 0,
            "max_nodes": 5,
        }
    ],
}

CONCURRENCY_PER_REPLICA = 32
NUM_PROMPTS_PER_REPLICA = 256
NUM_WARMUPS_PER_REPLICA = 0
REPLICA_COUNTS = [1, 2, 4, 8, 16, 20]

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
RESULTS_DIR = os.path.join(LLM_DIR, "results", "replica_sweep_throughput_optimized")


# ---------------------------------------------------------------------------
# Service lifecycle helpers
# ---------------------------------------------------------------------------

def generate_service_config(version_name, replicas):
    """Generate an Anyscale service YAML config dict."""
    version = VERSIONS[version_name]
    service_name = f"blog-llm-{version_name.replace('_', '-')}-{replicas}r"

    env_vars = {"NUM_REPLICAS": str(replicas)}
    env_vars.update(EXPERIMENT_B_ENV_VARS)

    config = {
        "name": service_name,
        "compute_config": COMPUTE_CONFIG,
        "env_vars": env_vars,
        "applications": [
            {
                "name": "default",
                "route_prefix": "/",
                "import_path": "app:app",
                "runtime_env": {
                    "working_dir": ".",
                },
            }
        ],
    }

    config["image_uri"] = version["image_uri"]

    return service_name, config


def deploy_service(service_name, config):
    """Write YAML config and deploy via the Anyscale CLI."""
    config_path = os.path.join(LLM_DIR, f".service-{service_name}.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    subprocess.run(
        ["anyscale", "service", "deploy", "-f", config_path],
        check=True,
        cwd=LLM_DIR,
    )
    return config_path


def wait_for_service(service_name, timeout=1800):
    """Block until the service reaches RUNNING and return (base_url, token)."""
    print(f"Waiting for service {service_name} ...")

    # Block until RUNNING (or raise on timeout / failure).
    subprocess.run(
        [
            "anyscale", "service", "wait",
            "--name", service_name,
            "--timeout-s", str(timeout),
        ],
        check=True,
    )

    # Fetch URL and auth token.
    result = subprocess.run(
        ["anyscale", "service", "status", "--name", service_name, "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    base_url = info["query_url"]
    token = info["query_auth_token"]
    print(f"  RUNNING — {base_url}")
    return base_url, token


def terminate_service(service_name):
    """Terminate an Anyscale service."""
    subprocess.run(
        ["anyscale", "service", "terminate", "--name", service_name],
        check=True,
    )
    print(f"Terminated {service_name}")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(base_url, token, replicas, version_name):
    """Run vllm bench serve against the deployed service."""
    result_path = os.path.join(
        RESULTS_DIR, f"{replicas}_replicas_{version_name}.json"
    )

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = token

    cmd = BENCH_CMD_BASE + [
        "--base-url",
        base_url,
        "--num-prompts",
        str(replicas * NUM_PROMPTS_PER_REPLICA),
        "--max-concurrency",
        str(replicas * CONCURRENCY_PER_REPLICA),
        "--num-warmups",
        str(replicas * NUM_WARMUPS_PER_REPLICA),
        "--result-filename",
        result_path,
    ]

    print(f"Running benchmark: {replicas} replicas, {version_name}")
    subprocess.run(cmd, check=True, env=env)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for version_name in VERSIONS:
        for replicas in REPLICA_COUNTS:
            print(f"\n{'=' * 60}")
            print(f"  Version: {version_name} | Replicas: {replicas}")
            print(f"{'=' * 60}\n")

            service_name, config = generate_service_config(
                version_name, replicas
            )

            # deploy_service(service_name, config)
            try:
                base_url, token = wait_for_service(service_name)
                run_benchmark(base_url, token, replicas, version_name)
            finally:
                try:
                    terminate_service(service_name)
                except Exception as e:
                    print(f"Warning: could not terminate {service_name}: {e}")

            # Cool-down between deployments
            time.sleep(30)

    print(f"\nSweep complete. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
