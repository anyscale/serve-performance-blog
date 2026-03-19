# Ray Serve Performance Benchmarks

Measuring the impact of Ray Serve nightly optimizations (HAProxy, gRPC, throughput mode, GC tuning) against the nightly baseline with all optimizations disabled.

## Comparison: Optimized vs Unoptimized

Both variants run the same nightly Ray image (`anyscale/ray-llm:nightly-py311-cu128`). The only difference is environment variables that toggle optimizations on or off.

| Flag | Optimized | Unoptimized |
|---|---|---|
| `RAY_SERVE_ENABLE_HA_PROXY` | 1 | unset |
| `RAY_SERVE_HAPROXY_TCP_NODELAY` | 1 | unset |
| `RAY_SERVE_USE_GRPC_BY_DEFAULT` | 1 | unset |
| `RAY_SERVE_THROUGHPUT_OPTIMIZED` | 1 | unset |
| `RAY_SERVE_RUN_ROUTER_IN_SEPARATE_LOOP` | 1 | unset |
| `RAY_SERVE_RUN_USER_CODE_IN_SEPARATE_THREAD` | 0 | unset |

---

## Benchmark Suites

### 1. Features — Isolated Optimization Microbenchmarks

Isolates individual optimizations using synthetic unary and streaming workloads to attribute gains to specific flags.

**Deployment architecture:**
- App: `unary_app/fastapi_dep.py` (unary) / `streaming_app/app.py` (streaming)
- Unary chain: `ASGIIngressDeployment` → `Echo` deployment (10KB payload echo)
- Streaming chain: `MyDeployment` → `GrandChildDeployment` (50 SSE chunks per request)
- `max_ongoing_requests`: 100,000 per deployment

**Cluster (1-replica configs):**
- Head: m8a.xlarge (4 vCPU, 16 GiB)
- Worker: 1x m7a.4xlarge (16 vCPU, 64 GiB)

**Cluster (8-replica configs):**
- Head: m8a.xlarge (4 vCPU, 16 GiB)
- Worker: 1x m7a.8xlarge

**Comparisons run:** gRPC on/off, HAProxy on/off, single event loop on/off, all-on/off (1 replica), all-on/off (8 replicas)

**Client:**
- Locust with 128 worker processes
- Unary concurrency sweep: 1, 2, 4, 8, 16, 32, 64, 128, 256
- Streaming concurrency sweep: 1, 2, 4, 8, 16, 32, 64, 128
- 60s per concurrency level, 15s cooldown between levels
- SLA thresholds: 200ms (unary), 700ms (streaming)

---

### 2. Recsys — Model Composition with Heterogeneous Compute

Two-tier DLRM recommendation pipeline exercising inter-deployment communication over heterogeneous hardware.

**Deployment architecture:**
- `IngressDeployment` (FastAPI): 2 CPU replicas, `max_ongoing_requests=1,000`
  - Synthesizes a batch of 32 feature vectors (13 dense, 26 sparse features)
  - Forwards batch to ranker via deployment handle
- `RankerDeployment`: 1 GPU replica, `max_ongoing_requests=1,000`
  - MiniDLRM model (embedding dim 64, cardinality 100K, bottom MLP [256, 64], top MLP [256, 128, 64])
  - `max_batch_size=100`, `batch_wait_timeout_s=0.05`

**Cluster:**
- Head: m8a.xlarge
- Worker: 1x g6.12xlarge (4x L4 GPU — only 1 GPU used by ranker)

**Client:**
- Locust, `constant(0)` wait time (max throughput)
- Concurrency sweep: 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500

---

### 3. LLM — Single-Node Replica Scaling + vLLM Comparison

LLM inference via `ray.serve.llm` APIs on a single node, plus standalone vLLM baseline (no Ray Serve).

**Deployment architecture:**
- Model: `openai/gpt-oss-20b`, TP=1, `max_model_len=4,096`
- `gpu_memory_utilization=0.95`, `max_num_seqs=256`, `max_num_batched_tokens=16,384`
- Replicas swept: [1, 2, 4, 8]
- Ingress replicas: `4 * num_replicas`
- `max_ongoing_requests=8,192`

**Cluster:**
- Head: p5.48xlarge

**Client:**
- `vllm bench serve`
- Input: 512 tokens, output: 128 tokens
- Concurrency per replica: 256
- Prompts per replica: 1,024

---


## Running Benchmarks

### Features

```bash
cd features
python sweep.py      # deploys all configs, benchmarks, collects results
python plot.py       # generates throughput-vs-latency plots
```

### Recsys

```bash
cd recsys
anyscale service deploy -f service-oss-nightly.yaml          # optimized
anyscale service deploy -f service-oss-nightly-unoptimized.yaml  # baseline

# Set env vars from Anyscale console
export HOST_OPTIMIZED="https://..."
export TOKEN_OPTIMIZED="..."
export HOST_UNOPTIMIZED="https://..."
export TOKEN_UNOPTIMIZED="..."

cd ..
python run_locust.py
python plot_recsys.py
```

### LLM

```bash
cd llm-single-node
python sweep_replicas.py
python plot_throughput.py
python plot_vllm_2x2.py
```

### Summary Chart

```bash
python plot_peak_gains.py    # aggregates peak gains across all suites
```

## Output

| Directory | Contents |
|---|---|
| `features/results/` | Per-comparison CSVs, per-concurrency JSONs, plots |
| `results/recsys/` | Locust JSON per concurrency level per variant |
| `llm-single-node/results/` | Ray Serve replica sweep results |
| `llm-single-node/results_vllm/` | Standalone vLLM baseline results |
| `peak_performance_gains.png` | Summary bar chart across all workloads |
