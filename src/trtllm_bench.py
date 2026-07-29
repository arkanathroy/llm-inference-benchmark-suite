"""Native TensorRT-LLM throughput/latency benchmark harness.

Runs inside envs/trtllm/venv, against a compiled engine produced by
src/build_trtllm_engine.py. TensorRT-LLM engines are invoked through
their own native Python runtime API (tensorrt_llm.runtime), since vLLM
does not execute compiled TRT engines directly.

Reads all hyperparameters from src/config.py's CONFIG singleton — see
that module for WHAT/WHY/EFFECT documentation on every value used here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer
from tensorrt_llm.runtime import ModelRunner

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONFIG
from gpu_monitor import GPUMonitor


def load_prompts(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def benchmark_batch_size(runner, tokenizer, prompts, batch_size, max_new_tokens,
                          num_measured_requests, num_warmup_requests):
    batch_prompts = (prompts * ((batch_size // len(prompts)) + 1))[:batch_size]
    input_ids = [tokenizer.encode(p, return_tensors="pt")[0] for p in batch_prompts]

    for _ in range(num_warmup_requests):
        runner.generate(batch_input_ids=input_ids, max_new_tokens=max_new_tokens)

    latencies_s = []
    total_tokens_generated = 0

    monitor = GPUMonitor()
    monitor.start()

    for _ in range(num_measured_requests):
        start = time.perf_counter()
        outputs = runner.generate(batch_input_ids=input_ids, max_new_tokens=max_new_tokens)
        elapsed = time.perf_counter() - start
        latencies_s.append(elapsed)
        for seq in outputs:
            total_tokens_generated += len(seq) - len(input_ids[0])

    gpu_stats = monitor.stop()

    total_time_s = sum(latencies_s)
    tokens_per_second = total_tokens_generated / total_time_s if total_time_s > 0 else 0.0
    avg_latency_s = total_time_s / len(latencies_s)

    return {
        "technique": "trtllm",
        "batch_size": batch_size,
        "avg_latency_s": avg_latency_s,
        "tokens_per_second": tokens_per_second,
        "num_measured_requests": num_measured_requests,
        **gpu_stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine_dir", required=True)
    parser.add_argument("--output", default="results/trtllm_benchmark.json")
    args = parser.parse_args()

    bench_cfg = CONFIG.benchmark
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.model.hf_repo)
    prompts = load_prompts(bench_cfg.prompts_file)

    runner = ModelRunner.from_dir(args.engine_dir)

    all_results = []
    for batch_size in bench_cfg.batch_sizes:
        print(f"[trtllm_bench] Benchmarking batch_size={batch_size} ...")
        result = benchmark_batch_size(
            runner=runner, tokenizer=tokenizer, prompts=prompts, batch_size=batch_size,
            max_new_tokens=bench_cfg.max_new_tokens,
            num_measured_requests=bench_cfg.num_measured_requests,
            num_warmup_requests=bench_cfg.num_warmup_requests,
        )
        all_results.append(result)
        print(json.dumps(result, indent=2, default=str))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"TensorRT-LLM benchmark results written to {output_path}")


if __name__ == "__main__":
    main()
