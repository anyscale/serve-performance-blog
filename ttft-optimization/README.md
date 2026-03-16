# TTFT Benchmark: Ray Serve vs vLLM Direct

Measures Time-To-First-Token (TTFT) and Time-Per-Output-Token (TPOT) across
three configurations to quantify Ray Serve's serving-layer overhead — and show
that optimizations eliminate it.

## Settings

| Label | Description | Env vars |
|-------|-------------|----------|
| **vLLM (direct)** | Standalone vLLM OpenAI server | — |
| **Ray Serve (optimized)** | Ray Serve with all optimizations on | `RAY_SERVE_ENABLE_HA_PROXY=1 RAY_SERVE_THROUGHPUT_OPTIMIZED=1` |
| **Ray Serve (default)** | Ray Serve without optimizations | `RAY_SERVE_ENABLE_HA_PROXY=0 RAY_SERVE_THROUGHPUT_OPTIMIZED=0` |

## Setup

- **Model:** `google/gemma-3-12b-it` with `tensor_parallel_size=4`
- **Workload:** random dataset, ISL=512, OSL=256
- **Concurrency levels:** 8, 64, 256
- **vLLM:** 0.15.0

## Reproduce

### 1. Start a server

**vLLM direct:**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-3-12b-it \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 --port 8000 \
    --disable-log-requests
```

**Ray Serve (optimized):**

```bash
RAY_SERVE_ENABLE_HA_PROXY=1 RAY_SERVE_THROUGHPUT_OPTIMIZED=1 \
    python serve_app.py
```

**Ray Serve (default):**

```bash
RAY_SERVE_ENABLE_HA_PROXY=0 RAY_SERVE_THROUGHPUT_OPTIMIZED=0 \
    python serve_app.py
```

### 2. Run the benchmark

```bash
python bench.py --endpoint http://localhost:8000/v1 \
    --model google/gemma-3-12b-it \
    --concurrencies 8,64,256 \
    --input-len 512 --output-len 256 \
    --results-dir results/vllm_direct
```

Num-prompts scales proportionally with concurrency (375 at c=8, 3000 at c=64,
12000 at c=256) to keep per-level runtime balanced.

Repeat for each setting, saving to a different `--results-dir`.

### 3. Plot

```bash
python plot.py --results-dirs \
    results/vllm_direct \
    results/ray_serve_optimized \
    results/ray_serve_default \
    --labels "vLLM (direct)" "Ray Serve w/ Optimizations" "Default Ray Serve"
```

Or run `python plot.py` with no arguments to regenerate the chart from the
reference data embedded in the script.
