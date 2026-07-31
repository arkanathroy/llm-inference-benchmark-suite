"""Unified benchmark client for any OpenAI-compatible server (vLLM or
llama.cpp). Handles the FP16, GPTQ, AWQ, and GGUF phases; TensorRT-LLM
is benchmarked separately through trtllm_bench.py's native runtime path
since it has no HTTP server in this design.

Reads all hyperparameters from src/config.py's CONFIG singleton — see
that module for WHAT/WHY/EFFECT documentation on every value used here.
Only --technique/--base_url/--model_id remain CLI args since those three
vary per invocation (which server is currently running), while everything
else is a fixed experiment-wide setting that belongs in config.py.

Logging: every print() below is also emitted as a structured JSON line via
the `log` module so a Promtail/Loki stack can tail this process's stdout
and Grafana can render a live per-request/per-batch progress view instead
of a silent multi-minute gap in the notebook cell output.
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
from log import get_logger

logger = get_logger("benchmark_runner")


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


def _progress_bar(current, total, width=30):
    filled = int(width * current / total) if total else width
    return "[" + "#" * filled + "-" * (width - filled) + f"] {current}/{total}"


def benchmark_technique(technique_name, base_url, model_id, prompts, batch_sizes,
                         max_new_tokens, num_warmup_requests, num_measured_requests):
    results = []
    num_batches = len(batch_sizes)

    with httpx.Client() as client:
        for batch_idx, batch_size in enumerate(batch_sizes, start=1):
            logger.info(
                f"batch {batch_idx}/{num_batches} starting "
                f"{_progress_bar(batch_idx - 1, num_batches)} batch_size={batch_size}",
                extra={"technique": technique_name, "phase": "batch_start",
                       "batch_size": batch_size, "batch_idx": batch_idx, "num_batches": num_batches},
            )

            warmup_start = time.perf_counter()
            for w in range(num_warmup_requests):
                single_request(client, base_url, model_id, prompts[0], max_new_tokens)
                logger.info(
                    f"batch_size={batch_size} warmup {w + 1}/{num_warmup_requests} done",
                    extra={"technique": technique_name, "phase": "warmup",
                           "batch_size": batch_size, "step": w + 1, "total": num_warmup_requests},
                )
            logger.info(
                f"batch_size={batch_size} warmup complete in {time.perf_counter() - warmup_start:.1f}s",
                extra={"technique": technique_name, "phase": "warmup_done", "batch_size": batch_size},
            )

            monitor = GPUMonitor()
            monitor.start()

            per_request_stats = []
            measure_start = time.perf_counter()
            for i in range(num_measured_requests):
                prompt = prompts[i % len(prompts)]
                stats = single_request(client, base_url, model_id, prompt, max_new_tokens)
                per_request_stats.append(stats)

                elapsed = time.perf_counter() - measure_start
                running_avg_tps = sum(s["tokens_per_second"] for s in per_request_stats) / len(per_request_stats)
                logger.info(
                    f"batch_size={batch_size} {_progress_bar(i + 1, num_measured_requests)} "
                    f"req_ttft={stats['ttft_s']:.3f}s req_tps={stats['tokens_per_second']:.1f} "
                    f"running_avg_tps={running_avg_tps:.1f} elapsed={elapsed:.1f}s",
                    extra={"technique": technique_name, "phase": "measuring",
                           "batch_size": batch_size, "step": i + 1, "total": num_measured_requests,
                           "ttft_s": stats["ttft_s"], "tokens_per_second": stats["tokens_per_second"],
                           "running_avg_tps": running_avg_tps},
                )

            gpu_stats = monitor.stop()

            avg_ttft = sum(s["ttft_s"] for s in per_request_stats) / len(per_request_stats)
            avg_tps = sum(s["tokens_per_second"] for s in per_request_stats) / len(per_request_stats)
            avg_latency = sum(s["total_latency_s"] for s in per_request_stats) / len(per_request_stats)

            logger.info(
                f"batch_size={batch_size} COMPLETE avg_ttft={avg_ttft:.3f}s avg_tps={avg_tps:.1f} "
                f"avg_latency={avg_latency:.3f}s peak_mem_mb={gpu_stats.get('peak_mem_used_mb')} "
                f"avg_util%={gpu_stats.get('avg_util_percent')}",
                extra={"technique": technique_name, "phase": "batch_done", "batch_size": batch_size,
                       "avg_ttft_s": avg_ttft, "avg_tokens_per_second": avg_tps,
                       "avg_latency_s": avg_latency, **gpu_stats},
            )

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

    logger.info(
        f"Starting benchmark for technique={args.technique} model={args.model_id} "
        f"base_url={args.base_url} batch_sizes={bench_cfg.batch_sizes} "
        f"num_warmup={bench_cfg.num_warmup_requests} num_measured={bench_cfg.num_measured_requests}",
        extra={"technique": args.technique, "phase": "start"},
    )

    overall_start = time.perf_counter()
    results = benchmark_technique(
        technique_name=args.technique, base_url=args.base_url, model_id=args.model_id,
        prompts=prompts, batch_sizes=bench_cfg.batch_sizes,
        max_new_tokens=bench_cfg.max_new_tokens,
        num_warmup_requests=bench_cfg.num_warmup_requests,
        num_measured_requests=bench_cfg.num_measured_requests,
    )

    csv_path = Path(CONFIG.output.results_dir) / CONFIG.output.csv_name
    append_results_csv(results, str(csv_path))

    logger.info(
        f"Benchmark for technique={args.technique} FINISHED in "
        f"{time.perf_counter() - overall_start:.1f}s. Appended {len(results)} rows to {csv_path}",
        extra={"technique": args.technique, "phase": "finished", "num_rows": len(results)},
    )
    print(f"Appended {len(results)} rows to {csv_path}")


if __name__ == "__main__":
    main()
