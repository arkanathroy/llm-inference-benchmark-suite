"""Unified benchmark client for any OpenAI-compatible server (vLLM or
llama.cpp). Handles the FP16, GPTQ, AWQ, and GGUF phases; TensorRT-LLM
is benchmarked separately through trtllm_bench.py's native runtime path
since it has no HTTP server in this design.

Reads all hyperparameters from src/config.py's CONFIG singleton — see
that module for WHAT/WHY/EFFECT documentation on every value used here.
Only --technique/--base_url/--model_id remain CLI args since those three
vary per invocation (which server is currently running), while everything
else is a fixed experiment-wide setting that belongs in config.py.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONFIG
from gpu_monitor import GPUMonitor


def load_prompts(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def single_request(client, base_url, model, prompt, max_new_tokens):
    start = time.perf_counter()
    first_token_time = None
    generated_tokens = 0

    with client.stream(
        "POST", f"{base_url}/completions",
        json={"model": model, "prompt": prompt, "max_tokens": max_new_tokens, "stream": True},
        timeout=120,
    ) as response:
        for line in response.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            if line.strip() == "data: [DONE]":
                break
            if first_token_time is None:
                first_token_time = time.perf_counter()
            generated_tokens += 1

    end = time.perf_counter()
    ttft_s = (first_token_time - start) if first_token_time else (end - start)
    total_s = end - start
    tokens_per_second = generated_tokens / total_s if total_s > 0 else 0.0

    return {
        "ttft_s": ttft_s,
        "total_latency_s": total_s,
        "generated_tokens": generated_tokens,
        "tokens_per_second": tokens_per_second,
    }


def benchmark_technique(technique_name, base_url, model_id, prompts, batch_sizes,
                         max_new_tokens, num_warmup_requests, num_measured_requests):
    results = []
    with httpx.Client() as client:
        for batch_size in batch_sizes:
            print(f"[benchmark_runner] {technique_name}: batch_size={batch_size}")

            for _ in range(num_warmup_requests):
                single_request(client, base_url, model_id, prompts[0], max_new_tokens)

            monitor = GPUMonitor()
            monitor.start()

            per_request_stats = []
            for i in range(num_measured_requests):
                prompt = prompts[i % len(prompts)]
                stats = single_request(client, base_url, model_id, prompt, max_new_tokens)
                per_request_stats.append(stats)

            gpu_stats = monitor.stop()

            avg_ttft = sum(s["ttft_s"] for s in per_request_stats) / len(per_request_stats)
            avg_tps = sum(s["tokens_per_second"] for s in per_request_stats) / len(per_request_stats)
            avg_latency = sum(s["total_latency_s"] for s in per_request_stats) / len(per_request_stats)

            results.append({
                "technique": technique_name,
                "batch_size": batch_size,
                "avg_ttft_s": avg_ttft,
                "avg_latency_s": avg_latency,
                "avg_tokens_per_second": avg_tps,
                "num_measured_requests": num_measured_requests,
                **gpu_stats,
            })

    return results


def append_results_csv(results, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    fieldnames = list(results[0].keys()) if results else []
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in results:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--technique", required=True)
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--model_id", required=True)
    args = parser.parse_args()

    bench_cfg = CONFIG.benchmark
    prompts = load_prompts(bench_cfg.prompts_file)

    results = benchmark_technique(
        technique_name=args.technique, base_url=args.base_url, model_id=args.model_id,
        prompts=prompts, batch_sizes=bench_cfg.batch_sizes,
        max_new_tokens=bench_cfg.max_new_tokens,
        num_warmup_requests=bench_cfg.num_warmup_requests,
        num_measured_requests=bench_cfg.num_measured_requests,
    )

    csv_path = Path(CONFIG.output.results_dir) / CONFIG.output.csv_name
    append_results_csv(results, str(csv_path))
    print(f"Appended {len(results)} rows to {csv_path}")


if __name__ == "__main__":
    main()
