"""Run vllm bench serve against a running endpoint at multiple concurrency levels.

Assumes the server is already running. Writes one JSON result file per
concurrency level into --results-dir. Num-prompts scales proportionally
with concurrency (12000 at c=256) to keep per-level runtime balanced.

Example:
    python bench.py --endpoint http://localhost:8000/v1 \
        --model google/gemma-3-12b-it \
        --concurrencies 8,64,256 \
        --input-len 512 --output-len 256 \
        --results-dir results/vllm_direct
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Scale num-prompts proportionally: 12000 prompts at concurrency 256.
BASE_CONCURRENCY = 256
BASE_NUM_PROMPTS = 12000


def num_prompts_for(concurrency: int) -> int:
    return max(1, int(BASE_NUM_PROMPTS * concurrency / BASE_CONCURRENCY))


def run_bench(
    base_url: str,
    model: str,
    concurrency: int,
    num_prompts: int,
    input_len: int,
    output_len: int,
    result_dir: Path,
) -> dict:
    result_dir.mkdir(parents=True, exist_ok=True)
    result_filename = f"c{concurrency}.json"

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main",
        "bench", "serve",
        "--base-url", base_url.rsplit("/v1", 1)[0],
        "--endpoint", "/v1/chat/completions",
        "--model", model,
        "--backend", "openai-chat",
        "--dataset-name", "random",
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(concurrency),
        "--input-len", str(input_len),
        "--output-len", str(output_len),
        "--save-result",
        "--result-dir", str(result_dir),
        "--result-filename", result_filename,
        "--ignore-eos",
        "--disable-tqdm",
        "--temperature", "0",
    ]

    print(f"  concurrency={concurrency} ...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        print(f"  FAILED (exit {proc.returncode})")
        print(proc.stderr[-2000:] if proc.stderr else "")
        raise RuntimeError(f"vllm bench serve failed at concurrency {concurrency}")

    return json.loads((result_dir / result_filename).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", required=True, help="Base URL, e.g. http://localhost:8000/v1")
    parser.add_argument("--model", default="google/gemma-3-12b-it")
    parser.add_argument("--concurrencies", default="8,64,256", help="Comma-separated concurrency levels")
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--results-dir", required=True, help="Directory to write result JSONs")
    args = parser.parse_args()

    concurrencies = [int(c) for c in args.concurrencies.split(",")]
    result_dir = Path(args.results_dir)

    print(f"Benchmarking {args.endpoint}")
    for c in concurrencies:
        n = num_prompts_for(c)
        print(f"  concurrency={c}, num_prompts={n}")
        run_bench(args.endpoint, args.model, c, n, args.input_len, args.output_len, result_dir)

    print(f"Done. Results in {result_dir}/")


if __name__ == "__main__":
    main()
