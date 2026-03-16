"""
Deploy the autoscaling nightly-optimized service.

This deploys a single service that autoscales based on num_ongoing_requests
(target=20 per replica) instead of pinning a fixed replica count.

Usage:
    cd serve-performance-blog/llm/scripts
    python deploy_autoscale.py
    python deploy_autoscale.py --min-replicas 1 --max-replicas 20 --target-ongoing-requests 20
"""

import argparse
import json
import os
import subprocess

import yaml


IMAGE_URI = "anyscale/ray-llm:nightly-py311-cu128"

ENV_VARS = {
    "RAY_SERVE_ENABLE_HA_PROXY": "1",
    "RAY_SERVE_HAPROXY_TCP_NODELAY": "1",
    "RAY_SERVE_RUN_SYNC_IN_THREADPOOL": "0",
    "RAY_SERVE_USE_GRPC_BY_DEFAULT": "1",
}

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_DIR = os.path.dirname(SCRIPT_DIR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-replicas", type=int, default=1)
    parser.add_argument("--max-replicas", type=int, default=20)
    parser.add_argument("--target-ongoing-requests", type=int, default=20)
    parser.add_argument(
        "--service-name",
        default="blog-llm-nightly-autoscale",
    )
    args = parser.parse_args()

    env_vars = {
        "MIN_REPLICAS": str(args.min_replicas),
        "MAX_REPLICAS": str(args.max_replicas),
        "TARGET_ONGOING_REQUESTS": str(args.target_ongoing_requests),
    }
    env_vars.update(ENV_VARS)

    config = {
        "name": args.service_name,
        "image_uri": IMAGE_URI,
        "compute_config": COMPUTE_CONFIG,
        "env_vars": env_vars,
        "applications": [
            {
                "name": "default",
                "route_prefix": "/",
                "import_path": "app_autoscale:app",
                "runtime_env": {
                    "working_dir": ".",
                },
            }
        ],
    }

    config_path = os.path.join(LLM_DIR, f".service-{args.service_name}.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Deploying {args.service_name} "
          f"(replicas: {args.min_replicas}-{args.max_replicas}, "
          f"target_ongoing_requests: {args.target_ongoing_requests})")

    subprocess.run(
        ["anyscale", "service", "deploy", "-f", config_path],
        check=True,
        cwd=LLM_DIR,
    )

    # Wait and print connection info.
    subprocess.run(
        ["anyscale", "service", "wait",
         "--name", args.service_name,
         "--timeout-s", "1800"],
        check=True,
    )

    result = subprocess.run(
        ["anyscale", "service", "status",
         "--name", args.service_name, "--json"],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(result.stdout)
    print(f"\nService RUNNING")
    print(f"  URL:   {info['query_url']}")
    print(f"  Token: {info['query_auth_token']}")


if __name__ == "__main__":
    main()
