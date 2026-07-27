"""
benchmark_runner.py
====================
Core latency/throughput measurement harness shared across every
precision variant and serving engine tested in this suite (raw
HuggingFace generate(), vLLM, and TensorRT-LLM).

Metrics collected per request, matching the JD's explicit ask for
"cost-per-token recommendations" and "performance cliffs":
  - TTFT (Time To First Token): latency from request submission to the
    first generated token becoming available. This is the metric most
    correlated with *perceived* responsiveness in interactive use
    (chat UIs, voice agents) -- a request with slow TTFT but fast
    subsequent tokens still feels sluggish to a human.
  - TPOT (Time Per Output Token): average inter-token latency after the
    first token, i.e. steady-state decode speed.
  - E2E latency: total wall-clock time for the full response.
  - Throughput: tokens/second aggregated across all concurrent requests
    at a given concurrency level.

WHY track TTFT and TPOT separately rather than only E2E latency: they
are dominated by different bottlenecks. TTFT is dominated by prefill
compute (scales with prompt length) and queueing delay under
concurrency. TPOT is dominated by decode-phase memory bandwidth (KV
cache read/write) and batching efficiency.
"""

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class RequestResult:
    request_id: int
    prompt_tokens: int
    output_tokens: int
    ttft_s: float
    e2e_s: float
    success: bool
    error: Optional[str] = None

    @property
    def tpot_s(self) -> float:
        if self.output_tokens <= 1:
            return 0.0
        return (self.e2e_s - self.ttft_s) / (self.output_tokens - 1)


@dataclass
class BenchmarkResult:
    engine: str
    precision: str
    concurrency: int
    requests: List[RequestResult] = field(default_factory=list)
    wall_clock_s: float = 0.0

    def _successful(self):
        return [r for r in self.requests if r.success]

    def percentile(self, values: List[float], p: float) -> float:
        if not values:
            return 0.0
        values_sorted = sorted(values)
        idx = int(len(values_sorted) * p / 100)
        idx = min(idx, len(values_sorted) - 1)
        return values_sorted[idx]

    def summary(self) -> dict:
        ok = self._successful()
        if not ok:
            return {"engine": self.engine, "precision": self.precision,
                    "concurrency": self.concurrency, "n_success": 0,
                    "n_failed": len(self.requests)}

        ttfts = [r.ttft_s for r in ok]
        tpots = [r.tpot_s for r in ok if r.output_tokens > 1]
        e2es = [r.e2e_s for r in ok]
        total_output_tokens = sum(r.output_tokens for r in ok)

        return {
            "engine": self.engine,
            "precision": self.precision,
            "concurrency": self.concurrency,
            "n_success": len(ok),
            "n_failed": len(self.requests) - len(ok),
            "ttft_p50_ms": self.percentile(ttfts, 50) * 1000,
            "ttft_p95_ms": self.percentile(ttfts, 95) * 1000,
            "ttft_p99_ms": self.percentile(ttfts, 99) * 1000,
            "tpot_p50_ms": self.percentile(tpots, 50) * 1000 if tpots else 0.0,
            "tpot_p95_ms": self.percentile(tpots, 95) * 1000 if tpots else 0.0,
            "e2e_p50_ms": self.percentile(e2es, 50) * 1000,
            "e2e_p99_ms": self.percentile(e2es, 99) * 1000,
            "aggregate_throughput_tok_s": total_output_tokens / self.wall_clock_s
            if self.wall_clock_s > 0 else 0.0,
        }


async def run_concurrent_benchmark(
    infer_fn: Callable,
    prompts: List[str],
    concurrency: int,
    engine_name: str,
    precision_name: str,
) -> BenchmarkResult:
    """
    Fires `concurrency` requests simultaneously via asyncio, each calling
    the engine-specific `infer_fn(prompt, request_id) -> RequestResult`.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_infer(prompt, req_id):
        async with semaphore:
            return await infer_fn(prompt, req_id)

    t0 = time.time()
    tasks = [
        bounded_infer(prompt, i)
        for i, prompt in enumerate(prompts)
    ]
    results = await asyncio.gather(*tasks)
    wall_clock = time.time() - t0

    return BenchmarkResult(
        engine=engine_name,
        precision=precision_name,
        concurrency=concurrency,
        requests=list(results),
        wall_clock_s=wall_clock,
    )


def estimate_cost_per_million_tokens(
    throughput_tok_s: float, instance_hourly_rate: float
) -> float:
    """
    cost per 1M tokens = (instance $/hr) / (tokens/sec x 3600 sec/hr) x 1,000,000
    """
    if throughput_tok_s <= 0:
        return float("inf")
    tokens_per_hour = throughput_tok_s * 3600
    return (instance_hourly_rate / tokens_per_hour) * 1_000_000
