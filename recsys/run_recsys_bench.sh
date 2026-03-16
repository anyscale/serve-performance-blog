#!/bin/bash
set -e

# Recsys benchmark: OSS nightly optimized vs unoptimized
# Requires: HOST_OPTIMIZED, TOKEN_OPTIMIZED, HOST_UNOPTIMIZED, TOKEN_UNOPTIMIZED env vars
#
# Deploy services first:
#   cd serve-perf/blog/recsys && anyscale service deploy -f service-oss-nightly.yaml
#   cd serve-perf/blog/recsys && anyscale service deploy -f service-oss-nightly-unoptimized.yaml

HOST_OPTIMIZED="${HOST_OPTIMIZED:-https://blog-recsys-oss-nightly-optimized-pyz23.cld-kvedzwag2qa8i5bj.s.anyscaleuserdata-staging.com}"
TOKEN_OPTIMIZED="${TOKEN_OPTIMIZED:-hzIbFcO7dZ1Z16VWvYGiSoIX2cUFZZjjU9gSCwj40gk}"
HOST_UNOPTIMIZED="${HOST_UNOPTIMIZED:-https://blog-recsys-oss-nightly-unoptimized-pyz23.cld-kvedzwag2qa8i5bj.s.anyscaleuserdata-staging.com}"
TOKEN_UNOPTIMIZED="${TOKEN_UNOPTIMIZED:-dlYwPiuNNTV5feRU0qOCM0XmNDvNGoumHKzJSs0yvXY}"

BLOG_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BLOG_DIR"

mkdir -p results/recsys

echo "=== Recsys benchmark: OSS Nightly Unoptimized ==="
python run_locust.py \
  -f recsys/locustfile.py \
  -o results/recsys \
  --host "$HOST_UNOPTIMIZED" \
  --token "$TOKEN_UNOPTIMIZED" \
  --route-prefix "/?user_id=1" \
  -t 60 \
  --name recsys-oss-nightly-unoptimized

echo "=== Recsys benchmark: OSS Nightly Optimized ==="
python run_locust.py \
  -f recsys/locustfile.py \
  -o results/recsys \
  --host "$HOST_OPTIMIZED" \
  --token "$TOKEN_OPTIMIZED" \
  --route-prefix "/?user_id=1" \
  -t 60 \
  --name recsys-oss-nightly-optimized

echo "Done. Plot with: python plot_recsys.py"
