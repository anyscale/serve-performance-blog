"""
Ray Serve LLM app entry point for Anyscale service deployment.

The replica count is read from the NUM_REPLICAS environment variable,
allowing the same app.py to be reused across service configs that only
differ in scale and Docker image.
"""

import os

from ray.serve.llm import (
    LLMConfig,
    LLMServingArgs,
    ModelLoadingConfig,
    build_openai_app,
)

num_replicas = int(os.environ.get("NUM_REPLICAS", "1"))
num_ingress_replicas = num_replicas * 4

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
            min_replicas=num_replicas,
            max_replicas=num_replicas,
        ),
        max_ongoing_requests=8192,
    ),
)

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
