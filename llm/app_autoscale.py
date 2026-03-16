"""
Ray Serve LLM app with autoscaling based on num_ongoing_requests.

Environment variables:
  - MIN_REPLICAS: minimum replica count (default: 1)
  - MAX_REPLICAS: maximum replica count (default: 20)
  - TARGET_ONGOING_REQUESTS: target num ongoing requests per replica (default: 20)
"""

import os

from ray.serve.llm import (
    LLMConfig,
    LLMServingArgs,
    ModelLoadingConfig,
    build_openai_app,
)

min_replicas = int(os.environ.get("MIN_REPLICAS", "1"))
max_replicas = int(os.environ.get("MAX_REPLICAS", "20"))
target_ongoing_requests = int(os.environ.get("TARGET_ONGOING_REQUESTS", "20"))

llm_config = LLMConfig(
    model_loading_config=ModelLoadingConfig(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        model_source="Qwen/Qwen2.5-0.5B-Instruct",
    ),
    engine_kwargs=dict(
        tensor_parallel_size=1,
        max_model_len=32000,
    ),
    deployment_config=dict(
        autoscaling_config=dict(
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            target_ongoing_requests=target_ongoing_requests,
        ),
        max_ongoing_requests=8192,
    ),
)

# Scale ingress replicas with max to ensure they're never the bottleneck.
num_ingress_replicas = max_replicas * 4

app = build_openai_app(
    LLMServingArgs(
        llm_configs=[llm_config],
        ingress_deployment_config={
            "autoscaling_config": {
                "min_replicas": num_ingress_replicas,
                "max_replicas": num_ingress_replicas,
            },
            "max_replicas_per_node": 16,
        },
    )
)
