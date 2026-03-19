vllm serve openai/gpt-oss-20b \
 --port 8000 \
 --tensor-parallel-size 1 \
 --max-model-len 4096 \
 --gpu-memory-utilization 0.95 \
 --max-num-seqs 256 \
 --max-num-batched-tokens 16384 \
 --no-enable-prefix-caching