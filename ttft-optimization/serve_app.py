"""Minimal Ray Serve + vLLM deployment for benchmarking.

Start with the appropriate env vars before running:

    RAY_SERVE_ENABLE_HA_PROXY=1 RAY_SERVE_THROUGHPUT_OPTIMIZED=1 python serve_app.py

Or without optimizations:

    RAY_SERVE_ENABLE_HA_PROXY=0 RAY_SERVE_THROUGHPUT_OPTIMIZED=0 python serve_app.py
"""

import os

import ray
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app
from ray.serve.schema import LoggingConfig

MODEL_ID = "google/gemma-3-12b-it"
PORT = 8000

QUIET_ENV = {
    "env_vars": {
        "VLLM_LOGGING_LEVEL": "WARNING",
        "RAY_SERVE_LOG_TO_STDERR": "0",
    },
}

llm_config = LLMConfig(
    model_loading_config=dict(model_id=MODEL_ID, model_source=MODEL_ID),
    deployment_config=dict(autoscaling_config=dict(min_replicas=1, max_replicas=1)),
    engine_kwargs=dict(tensor_parallel_size=4),
    runtime_env=QUIET_ENV,
)

app = build_openai_app(dict(llm_configs=[llm_config]))

if __name__ == "__main__":
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    ray.init(ignore_reinit_error=True)
    serve.start(http_options={"port": PORT})
    serve.run(
        app,
        name="default",
        blocking=True,
        logging_config=LoggingConfig(log_level="WARNING", enable_access_log=False),
    )
