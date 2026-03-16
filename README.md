# Blog Post Benchmarks: Ray Serve Performance

Reproduction scripts and results for the Ray Serve performance blog post.
Compares **Ray Serve nightly (optimized)** vs **Ray Serve nightly (default)**.

## Benchmarks

### 1. Feature-by-Feature Microbenchmarks (`features/`)

Isolates individual optimizations (HAProxy, gRPC, GC/event-loop tuning, all-on)
using a synthetic 10 KB unary/streaming workload. Sweeps concurrency from 1–256.

- **Unary app:** `unary_app/fastapi_dep.py` — echo deployment
- **Streaming app:** `streaming_app/app.py` — mock streaming deployment
- **Configs:** `features/configs/` — one YAML per (feature × protocol) combination
- **Run:** `python features/sweep.py` (calls `anyscale service deploy`; # serve run <config>.yaml)
- **Plot:** `python features/plot.py`

### 2. Recsys — Model Composition (`recsys/`)

Two-tier DLRM recommendation system: CPU ingress + GPU ranker.
Measures RPS and latency at increasing concurrency, showing HAProxy + gRPC impact.

```bash
cd recsys
anyscale service deploy -f service-oss-nightly.yaml              # serve run service-oss-nightly.yaml
anyscale service deploy -f service-oss-nightly-unoptimized.yaml  # serve run service-oss-nightly-unoptimized.yaml
# Set HOST_*/TOKEN_* env vars, then:
cd ..
python run_locust.py ...
python plot_recsys.py
```

### 3. LLM — Streaming Inference (`llm/`)

Real LLM inference using `ray.serve.llm` APIs.
Replica sweep (1–20 replicas) with `vllm bench serve`.

```bash
cd llm/scripts
python sweep_replicas.py             # generates YAMLs + anyscale service deploy; # serve run <generated>.yaml
python visualize_replica_sweep.py
```

### 4. TTFT Optimization (`ttft-optimization/`)

Compares Time-To-First-Token across vLLM direct, Ray Serve optimized,
and default Ray Serve (gemma-3-12b-it, TP=4, ISL=512, OSL=256).

```bash
# Start server (vLLM direct, or Ray Serve via serve_app.py)
# serve run ttft-optimization/serve_app.py
python ttft-optimization/bench.py --endpoint http://localhost:8000/v1 \
    --model google/gemma-3-12b-it --concurrencies 8,64,256 \
    --results-dir results/vllm_direct
python ttft-optimization/plot.py
```

## Project Structure

```
serve-performance-blog/
├── README.md
├── features/
│   ├── configs/                  # Service YAMLs per feature × protocol
│   ├── locustfile_10kb.py
│   ├── sweep.py
│   ├── plot.py
│   └── results/
├── unary_app/
│   └── fastapi_dep.py            # Echo deployment (used by features/ unary configs)
├── streaming_app/
│   ├── app.py                    # Mock streaming deployment (used by features/ streaming configs)
│   └── llm_stream_benchmark.py   # Streaming benchmark script (used by features/sweep.py)
├── recsys/
│   ├── app.py, model.py, config.py
│   ├── locustfile.py
│   ├── service-oss-nightly.yaml
│   └── service-oss-nightly-unoptimized.yaml
├── llm/
│   ├── app.py, app_autoscale.py
│   ├── scripts/
│   │   ├── sweep_replicas.py
│   │   ├── sweep_concurrency.py
│   │   ├── visualize_replica_sweep.py
│   │   └── visualize_rps_sweep.py
│   └── results/
├── ttft-optimization/
│   ├── README.md
│   ├── bench.py
│   ├── plot.py
│   └── serve_app.py
├── results/                      # Recsys result JSONs
├── run_locust.py                 # Locust runner
├── plot_recsys.py
├── plot_llm.py
├── plot_peak_gains.py
└── *.png                         # Pre-generated plots
```
