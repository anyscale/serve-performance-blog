import argparse
import random
import httpx
import asyncio
import time
import numpy as np
import csv
import os
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Dict
from tqdm.asyncio import tqdm
from datetime import datetime
import pandas as pd
from functools import partial


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mt", "--max-tokens", type=int, default=128)
    parser.add_argument("-tpot", type=float, default=0.005)
    parser.add_argument("-ttft", type=float, default=0.1)
    parser.add_argument("--host", type=str, default="http://0.0.0.0:30000")
    parser.add_argument("--path", type=str, default="/streaming")
    parser.add_argument("--output-csv", type=str, default="benchmark_results.csv", help="CSV file to save results")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of worker processes to use for load testing")
    parser.add_argument("--token", type=str, default="", help="Auth bearer token")
    parser.add_argument("--concurrencies", type=str, default="", help="Comma-separated concurrency levels (default: 64,128,256,512,1024,2048,4096,8192)")
    parser.add_argument("--request-multiplier", type=int, default=1, help="Multiply num_requests per concurrency level (useful for multi-replica setups)")
    return parser.parse_args()


@dataclass
class RequestResult:
    ttft: float
    tpots: List[float]
    tokens: List[str]
    end_to_end_latency: float

    @property
    def avg_tpot(self) -> float:
        return np.mean(self.tpots) if self.tpots else 0

    @property
    def tokens_per_sec(self) -> float:
        """Calculate tokens per second based on total tokens and end-to-end latency"""
        if self.end_to_end_latency > 0 and len(self.tokens) > 0:
            return len(self.tokens) / self.end_to_end_latency
        return 0


async def _request(semaphore: asyncio.Semaphore, client: httpx.AsyncClient, max_tokens: int, tpot: float, ttft: float, host: str, path: str, token: str = "") -> RequestResult:
    async with semaphore:
        headers = {
            "Connection": "keep-alive",
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        # Retry configuration
        max_retries = 3
        base_delay = 0.1  # Base delay in seconds
        max_delay = 2.0   # Maximum delay in seconds

        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                start_time = time.perf_counter()

                # Send the request with streaming enabled
                async with client.stream("GET", f"{host}{path}", params={"num_tokens": max_tokens, "tpot": tpot, "ttft": ttft, "message": "Hello, world!"}, headers=headers) as response:
                    # Check for 5xx errors and retry if needed
                    if response.status_code >= 500:
                        if attempt < max_retries:
                            # Calculate exponential backoff delay with jitter
                            delay = min(base_delay * (2 ** attempt), max_delay)
                            jitter = random.uniform(0, 0.1)  # Add small random jitter
                            total_delay = delay + jitter

                            print(f"5xx error (status {response.status_code}), retrying in {total_delay:.2f}s (attempt {attempt + 1}/{max_retries + 1})")
                            await asyncio.sleep(total_delay)
                            continue
                        else:
                            # Final attempt failed, raise the error
                            response.raise_for_status()

                    # For non-5xx errors, raise immediately (don't retry 4xx errors)
                    response.raise_for_status()

                    count = 0
                    first_token = True
                    tpots = []
                    tokens = []
                    ttft_actual = 0
                    last_token_time = None

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            token = line[6:].strip()  # Remove "data: " prefix and any whitespace
                            # Skip the [DONE] token for timing calculations
                            if token == "[DONE]":
                                continue

                            tokens.append(token)
                            current_time = time.perf_counter()

                            if first_token:
                                ttft_actual = current_time - start_time
                                first_token = False
                                last_token_time = current_time
                            else:
                                # Calculate TPOT as time between this token and the previous token
                                if last_token_time is not None:
                                    tpot_individual = current_time - last_token_time
                                    tpots.append(tpot_individual)
                                last_token_time = current_time
                            count += 1

                    end_time = time.perf_counter()

                    # Calculate end-to-end latency (total time from request start to completion)
                    end_to_end_latency = end_time - start_time

                    return RequestResult(ttft=ttft_actual, tpots=tpots, tokens=tokens, end_to_end_latency=end_to_end_latency)

            except httpx.HTTPStatusError as e:
                # If it's a 5xx error and we have retries left, continue to next attempt
                if e.response.status_code >= 500 and attempt < max_retries:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, 0.1)
                    total_delay = delay + jitter

                    print(f"5xx error (status {e.response.status_code}), retrying in {total_delay:.2f}s (attempt {attempt + 1}/{max_retries + 1})")
                    await asyncio.sleep(total_delay)
                    continue
                else:
                    # Re-raise for 4xx errors or when out of retries
                    raise
            except (httpx.RequestError, httpx.TimeoutException) as e:
                # For network errors, also retry
                if attempt < max_retries:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, 0.1)
                    total_delay = delay + jitter

                    print(f"Network error ({type(e).__name__}), retrying in {total_delay:.2f}s (attempt {attempt + 1}/{max_retries + 1})")
                    await asyncio.sleep(total_delay)
                    continue
                else:
                    # Re-raise when out of retries
                    raise


async def benchmark(concurrency: int, num_requests: int, pargs):
    print(f"Running benchmark with {num_requests} requests, concurrency {concurrency}")
    print(f"Target max_tokens={pargs.max_tokens}, tpot={pargs.tpot}, ttft={pargs.ttft}")

    # Track overall benchmark timing for system-level throughput
    benchmark_start_time = time.perf_counter()

    semaphore = asyncio.Semaphore(concurrency)

    # Create a shared HTTP client with settings that avoid connection reuse issues
    async with httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=concurrency + concurrency // 2,          # One connection per concurrent request
            keepalive_expiry=0,
        ),
        timeout=httpx.Timeout(
            # For streaming, a global timeout can be counterproductive; disable it
            timeout=None,
            # Keep connect snappy; if you hit connect timeouts, your server is saturated
            connect=10.0,
            # **Critical for streaming**: disable read timeout so long gaps don’t kill streams
            read=None,
            write=10.0,
            # Give the pool time to hand you a connection when all are busy
            pool=30.0
        ),
    ) as client:
        # Create tasks but don't await them yet
        tasks = [_request(semaphore, client, pargs.max_tokens, pargs.tpot, pargs.ttft, pargs.host, pargs.path, getattr(pargs, 'token', ''))
                 for _ in range(num_requests)]

        # Use tqdm to track progress of completing tasks
        results = []
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                        desc="Processing requests", unit="req"):
            result = await task
            results.append(result)

    # Calculate overall benchmark duration for system-level throughput
    benchmark_end_time = time.perf_counter()
    benchmark_duration = benchmark_end_time - benchmark_start_time

    # Calculate system-level metrics
    total_tokens = sum(len(result.tokens) for result in results)
    system_tokens_per_sec = total_tokens / benchmark_duration if benchmark_duration > 0 else 0

    # Extract all TTFTs, TPOTs, end-to-end latencies, and tokens per second for analysis
    ttfts = [result.ttft for result in results]
    all_tpots = [tpot for result in results for tpot in result.tpots]
    end_to_end_latencies = [result.end_to_end_latency for result in results]
    tokens_per_sec = [result.tokens_per_sec for result in results]

    # Calculate statistics
    avg_ttft = np.mean(ttfts)
    p50_ttft = np.percentile(ttfts, 50)
    p95_ttft = np.percentile(ttfts, 95)
    p99_ttft = np.percentile(ttfts, 99)

    avg_tpot = np.mean(all_tpots)
    p50_tpot = np.percentile(all_tpots, 50)
    p95_tpot = np.percentile(all_tpots, 95)
    p99_tpot = np.percentile(all_tpots, 99)

    avg_e2e = np.mean(end_to_end_latencies)
    p50_e2e = np.percentile(end_to_end_latencies, 50)
    p95_e2e = np.percentile(end_to_end_latencies, 95)
    p99_e2e = np.percentile(end_to_end_latencies, 99)

    avg_tokens_per_sec = np.mean(tokens_per_sec)
    p50_tokens_per_sec = np.percentile(tokens_per_sec, 50)
    p95_tokens_per_sec = np.percentile(tokens_per_sec, 95)
    p99_tokens_per_sec = np.percentile(tokens_per_sec, 99)

    # Print results in a nice table format
    table_width = 110
    print("\n" + "=" * table_width)
    title = f" Benchmark Results (Target TTFT={pargs.ttft:.3f}s, TPOT={pargs.tpot:.3f}s, Concurrency={concurrency}) "
    print(f"{title:^{table_width}}")
    print("=" * table_width)

    # Print system-level summary first
    print(f"System Throughput: {system_tokens_per_sec:.2f} tokens/sec")
    print(f"Total Tokens: {total_tokens:,} | Benchmark Duration: {benchmark_duration:.2f}s | Requests: {num_requests}")
    print("=" * table_width)

    # Table headers with proper spacing for longer values
    print(f"{'Metric':<10} | {'Average':<20} | {'P50':<20} | {'P95':<20} | {'P99':<20}")
    print(f"{'-'*10}-+-{'-'*20}-+-{'-'*20}-+-{'-'*20}-+-{'-'*20}")

    # TTFT row with better formatting
    ttft_avg = f"{avg_ttft:.6f} ({avg_ttft/pargs.ttft:.2f}x)"
    ttft_p50 = f"{p50_ttft:.6f} ({p50_ttft/pargs.ttft:.2f}x)"
    ttft_p95 = f"{p95_ttft:.6f} ({p95_ttft/pargs.ttft:.2f}x)"
    ttft_p99 = f"{p99_ttft:.6f} ({p99_ttft/pargs.ttft:.2f}x)"
    print(f"{'TTFT (s)':<10} | {ttft_avg:<20} | {ttft_p50:<20} | {ttft_p95:<20} | {ttft_p99:<20}")

    # TPOT row with better formatting
    tpot_avg = f"{avg_tpot:.6f} ({avg_tpot/pargs.tpot:.2f}x)"
    tpot_p50 = f"{p50_tpot:.6f} ({p50_tpot/pargs.tpot:.2f}x)"
    tpot_p95 = f"{p95_tpot:.6f} ({p95_tpot/pargs.tpot:.2f}x)"
    tpot_p99 = f"{p99_tpot:.6f} ({p99_tpot/pargs.tpot:.2f}x)"
    print(f"{'TPOT (s)':<10} | {tpot_avg:<20} | {tpot_p50:<20} | {tpot_p95:<20} | {tpot_p99:<20}")

    # End-to-end latency row
    e2e_avg = f"{avg_e2e:.6f}s"
    e2e_p50 = f"{p50_e2e:.6f}s"
    e2e_p95 = f"{p95_e2e:.6f}s"
    e2e_p99 = f"{p99_e2e:.6f}s"
    print(f"{'E2E (s)':<10} | {e2e_avg:<20} | {e2e_p50:<20} | {e2e_p95:<20} | {e2e_p99:<20}")

    # Tokens per second row
    tps_avg = f"{avg_tokens_per_sec:.2f} tok/s"
    tps_p50 = f"{p50_tokens_per_sec:.2f} tok/s"
    tps_p95 = f"{p95_tokens_per_sec:.2f} tok/s"
    tps_p99 = f"{p99_tokens_per_sec:.2f} tok/s"
    print(f"{'Tok/s':<10} | {tps_avg:<20} | {tps_p50:<20} | {tps_p95:<20} | {tps_p99:<20}")

    print("=" * table_width)

    return {
        "ttft_avg": avg_ttft,
        "tpot_avg": avg_tpot,
        "e2e_avg": avg_e2e,
        "tokens_per_sec_avg": avg_tokens_per_sec,
        "system_tokens_per_sec": system_tokens_per_sec,
        "total_tokens": total_tokens,
        "benchmark_duration": benchmark_duration,
        "ttft_p50": p50_ttft,
        "tpot_p50": p50_tpot,
        "e2e_p50": p50_e2e,
        "tokens_per_sec_p50": p50_tokens_per_sec,
        "ttft_p95": p95_ttft,
        "tpot_p95": p95_tpot,
        "e2e_p95": p95_e2e,
        "tokens_per_sec_p95": p95_tokens_per_sec,
        "ttft_p99": p99_ttft,
        "tpot_p99": p99_tpot,
        "e2e_p99": p99_e2e,
        "tokens_per_sec_p99": p99_tokens_per_sec,
        "num_requests": num_requests,
        "concurrency": concurrency,
        "max_tokens": pargs.max_tokens,
        "tpot": pargs.tpot,
        "ttft": pargs.ttft,
        "host": pargs.host,
        "path": pargs.path,
        "num_workers": 1,  # Single process benchmark always uses 1 worker
    }


def worker_process(worker_id: int, num_requests: int, concurrency: int, pargs):
    """Worker process function that runs benchmark for a subset of requests"""
    print(f"Worker {worker_id}: Starting benchmark with {num_requests} requests, concurrency {concurrency}")

    # Run the async benchmark in this process
    try:
        result = asyncio.run(benchmark(concurrency, num_requests, pargs))
        result['worker_id'] = worker_id
        print(f"Worker {worker_id}: Completed benchmark")
        return result
    except Exception as e:
        print(f"Worker {worker_id}: Failed with error: {e}")
        raise e


def benchmark_multiprocess(total_concurrency: int, total_requests: int, num_workers: int, pargs):
    """Run benchmark using multiprocessing when num_workers > 1"""
    print(f"Running multiprocess benchmark with {num_workers} workers")
    print(f"Total requests: {total_requests}, Total concurrency: {total_concurrency}")

    # Calculate requests and concurrency per worker
    requests_per_worker = total_requests // num_workers
    remaining_requests = total_requests % num_workers
    concurrency_per_worker = total_concurrency // num_workers
    remaining_concurrency = total_concurrency % num_workers

    # Create arguments for each worker
    worker_args = []
    for worker_id in range(num_workers):
        # Distribute remaining requests and concurrency to first few workers
        worker_requests = requests_per_worker + (1 if worker_id < remaining_requests else 0)
        worker_concurrency = concurrency_per_worker + (1 if worker_id < remaining_concurrency else 0)
        worker_args.append((worker_id, worker_requests, worker_concurrency, pargs))

    print(f"Worker distribution:")
    for i, (worker_id, requests, concurrency, _) in enumerate(worker_args):
        print(f"  Worker {worker_id}: {requests} requests, concurrency {concurrency}")

    # Track overall benchmark timing
    benchmark_start_time = time.perf_counter()

    # Start all worker processes
    with mp.Pool(processes=num_workers) as pool:
        # Use partial to unpack arguments
        worker_func = partial(worker_process)
        results = pool.starmap(worker_func, worker_args)

    benchmark_end_time = time.perf_counter()
    benchmark_duration = benchmark_end_time - benchmark_start_time

    # Aggregate results from all workers
    return aggregate_worker_results(results, benchmark_duration, total_concurrency, total_requests, pargs)


def aggregate_worker_results(worker_results: List[Dict], total_benchmark_duration: float,
                           total_concurrency: int, total_requests: int, pargs) -> Dict:
    """Aggregate results from multiple worker processes"""
    if not worker_results:
        raise ValueError("No worker results to aggregate")

    # Collect all individual metrics
    all_ttfts = []
    all_tpots = []
    all_e2es = []
    all_tokens_per_sec = []
    total_tokens = 0

    # Extract metrics from each worker result
    for result in worker_results:
        # For aggregation, we need to reconstruct individual request data
        # Since we only have aggregated stats, we'll use the averages multiplied by request count
        num_requests = result['num_requests']

        # Approximate individual values from averages (this is a limitation of the current structure)
        # In a real implementation, you might want to modify the worker to return raw data
        all_ttfts.extend([result['ttft_avg']] * num_requests)  # This is an approximation
        all_e2es.extend([result['e2e_avg']] * num_requests)   # This is an approximation
        all_tokens_per_sec.extend([result['tokens_per_sec_avg']] * num_requests)  # This is an approximation

        total_tokens += result['total_tokens']

    # Calculate system-level throughput
    system_tokens_per_sec = total_tokens / total_benchmark_duration if total_benchmark_duration > 0 else 0

    # Calculate aggregated statistics
    avg_ttft = np.mean(all_ttfts)
    p50_ttft = np.percentile(all_ttfts, 50)
    p95_ttft = np.percentile(all_ttfts, 95)
    p99_ttft = np.percentile(all_ttfts, 99)

    # For TPOT, we need to aggregate differently since it's token-to-token timing
    # Use weighted average based on total tokens from each worker
    total_worker_tokens = sum(result['total_tokens'] for result in worker_results)
    if total_worker_tokens > 0:
        avg_tpot = sum(result['tpot_avg'] * result['total_tokens'] for result in worker_results) / total_worker_tokens
        # For percentiles, use weighted approach or simple average (approximation)
        p50_tpot = np.mean([result['tpot_p50'] for result in worker_results])
        p95_tpot = np.mean([result['tpot_p95'] for result in worker_results])
        p99_tpot = np.mean([result['tpot_p99'] for result in worker_results])
    else:
        avg_tpot = p50_tpot = p95_tpot = p99_tpot = 0

    avg_e2e = np.mean(all_e2es)
    p50_e2e = np.percentile(all_e2es, 50)
    p95_e2e = np.percentile(all_e2es, 95)
    p99_e2e = np.percentile(all_e2es, 99)

    avg_tokens_per_sec = np.mean(all_tokens_per_sec)
    p50_tokens_per_sec = np.percentile(all_tokens_per_sec, 50)
    p95_tokens_per_sec = np.percentile(all_tokens_per_sec, 95)
    p99_tokens_per_sec = np.percentile(all_tokens_per_sec, 99)

    # Print aggregated results
    table_width = 110
    print("\n" + "=" * table_width)
    title = f" Multiprocess Benchmark Results (Workers={len(worker_results)}, Total Concurrency={total_concurrency}) "
    print(f"{title:^{table_width}}")
    print("=" * table_width)

    # Print system-level summary
    print(f"System Throughput: {system_tokens_per_sec:.2f} tokens/sec")
    print(f"Total Tokens: {total_tokens:,} | Benchmark Duration: {total_benchmark_duration:.2f}s | Requests: {total_requests}")
    print("=" * table_width)

    # Table headers with proper spacing
    print(f"{'Metric':<10} | {'Average':<20} | {'P50':<20} | {'P95':<20} | {'P99':<20}")
    print(f"{'-'*10}-+-{'-'*20}-+-{'-'*20}-+-{'-'*20}-+-{'-'*20}")

    # TTFT row
    ttft_avg = f"{avg_ttft:.6f} ({avg_ttft/pargs.ttft:.2f}x)"
    ttft_p50 = f"{p50_ttft:.6f} ({p50_ttft/pargs.ttft:.2f}x)"
    ttft_p95 = f"{p95_ttft:.6f} ({p95_ttft/pargs.ttft:.2f}x)"
    ttft_p99 = f"{p99_ttft:.6f} ({p99_ttft/pargs.ttft:.2f}x)"
    print(f"{'TTFT (s)':<10} | {ttft_avg:<20} | {ttft_p50:<20} | {ttft_p95:<20} | {ttft_p99:<20}")

    # TPOT row
    tpot_avg = f"{avg_tpot:.6f} ({avg_tpot/pargs.tpot:.2f}x)"
    tpot_p50 = f"{p50_tpot:.6f} ({p50_tpot/pargs.tpot:.2f}x)"
    tpot_p95 = f"{p95_tpot:.6f} ({p95_tpot/pargs.tpot:.2f}x)"
    tpot_p99 = f"{p99_tpot:.6f} ({p99_tpot/pargs.tpot:.2f}x)"
    print(f"{'TPOT (s)':<10} | {tpot_avg:<20} | {tpot_p50:<20} | {tpot_p95:<20} | {tpot_p99:<20}")

    # End-to-end latency row
    e2e_avg = f"{avg_e2e:.6f}s"
    e2e_p50 = f"{p50_e2e:.6f}s"
    e2e_p95 = f"{p95_e2e:.6f}s"
    e2e_p99 = f"{p99_e2e:.6f}s"
    print(f"{'E2E (s)':<10} | {e2e_avg:<20} | {e2e_p50:<20} | {e2e_p95:<20} | {e2e_p99:<20}")

    # Tokens per second row
    tps_avg = f"{avg_tokens_per_sec:.2f} tok/s"
    tps_p50 = f"{p50_tokens_per_sec:.2f} tok/s"
    tps_p95 = f"{p95_tokens_per_sec:.2f} tok/s"
    tps_p99 = f"{p99_tokens_per_sec:.2f} tok/s"
    print(f"{'Tok/s':<10} | {tps_avg:<20} | {tps_p50:<20} | {tps_p95:<20} | {tps_p99:<20}")

    print("=" * table_width)

    return {
        "ttft_avg": avg_ttft,
        "tpot_avg": avg_tpot,
        "e2e_avg": avg_e2e,
        "tokens_per_sec_avg": avg_tokens_per_sec,
        "system_tokens_per_sec": system_tokens_per_sec,
        "total_tokens": total_tokens,
        "benchmark_duration": total_benchmark_duration,
        "ttft_p50": p50_ttft,
        "tpot_p50": p50_tpot,
        "e2e_p50": p50_e2e,
        "tokens_per_sec_p50": p50_tokens_per_sec,
        "ttft_p95": p95_ttft,
        "tpot_p95": p95_tpot,
        "e2e_p95": p95_e2e,
        "tokens_per_sec_p95": p95_tokens_per_sec,
        "ttft_p99": p99_ttft,
        "tpot_p99": p99_tpot,
        "e2e_p99": p99_e2e,
        "tokens_per_sec_p99": p99_tokens_per_sec,
        "num_requests": total_requests,
        "concurrency": total_concurrency,
        "max_tokens": pargs.max_tokens,
        "tpot": pargs.tpot,
        "ttft": pargs.ttft,
        "host": pargs.host,
        "path": pargs.path,
        "num_workers": len(worker_results),
    }

def save_results_to_csv(all_results: List[Dict], output_file: str):
    """Save benchmark results to CSV file"""
    if not all_results:
        return

    # Get fieldnames from the first result
    fieldnames = list(all_results[0].keys())

    # Add timestamp column
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for result in all_results:
        result['timestamp'] = timestamp

    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False)

    print(f"\nResults saved to {output_file}")


def run_concurrency_sweep(pargs):
    """Run benchmark across multiple concurrency levels"""
    if pargs.concurrencies:
        concurrency_levels = [int(c) for c in pargs.concurrencies.split(",")]
    else:
        concurrency_levels = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    all_results = []

    print(f"Starting concurrency sweep with levels: {concurrency_levels}")
    print(f"Each run will have num_requests = 4 * concurrency")
    print("=" * 150)

    for concurrency in concurrency_levels:
        num_requests = 60 * concurrency * pargs.request_multiplier
        print(f"\n🚀 Running concurrency level: {concurrency} (requests: {num_requests})")

        try:
            if pargs.num_workers > 1:
                print(f"Using multiprocessing with {pargs.num_workers} workers")
                result = benchmark_multiprocess(concurrency, num_requests, pargs.num_workers, pargs)
            else:
                print("Using single-process async benchmark")
                result = asyncio.run(benchmark(concurrency, num_requests, pargs))
            all_results.append(result)
            print(f"✅ Completed concurrency level: {concurrency}")
        except Exception as e:
            print(f"❌ Failed concurrency level {concurrency}: {e}")
            raise e

    # Save all results to CSV
    if all_results:
        save_results_to_csv(all_results, pargs.output_csv)

        # Print summary
        print("\n" + "=" * 150)
        print("📊 CONCURRENCY SWEEP SUMMARY")
        print("=" * 150)
        print(f"{'Concurrency':<12} | {'Requests':<10} | {'Workers':<8} | {'System Tok/s':<14} | {'Total Tokens':<12} | {'Duration':<10} | {'TTFT Avg':<12} | {'TPOT Avg':<12} | {'E2E Avg':<12}")
        print("-" * 158)

        for result in all_results:
            workers = result.get('num_workers', 1)
            print(f"{result['concurrency']:<12} | {result['num_requests']:<10} | {workers:<8} | {result['system_tokens_per_sec']:<14.2f} | {result['total_tokens']:<12,} | {result['benchmark_duration']:<10.2f} | {result['ttft_avg']:<12.6f} | {result['tpot_avg']:<12.6f} | {result['e2e_avg']:<12.6f}")

        print("=" * 150)
        print(f"🎉 Concurrency sweep completed! Results saved to: {pargs.output_csv}")
    else:
        print("⚠️  No successful benchmark runs completed.")


def main(pargs):
    run_concurrency_sweep(pargs)


if __name__ == "__main__":
    # Required for multiprocessing on Windows and some Unix systems
    mp.set_start_method('spawn', force=True)
    pargs = _parse_args()
    main(pargs)
