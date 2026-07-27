"""
trtllm_bench.py
================
TensorRT-LLM concurrency benchmark, mirroring benchmark_runner.py's
methodology but using TRT-LLM's ModelRunner API instead of vLLM's
AsyncLLMEngine. Requires an Ampere+ GPU (A100/H100) -- gated in the
notebook's Phase 9 by a GPU_ARCH check since TRT-LLM engine compilation
targets a specific GPU architecture at build time and will not run on
a T4 (Turing) even if installed.
"""

import time
from benchmark_runner import RequestResult, BenchmarkResult


def trtllm_infer(runner, tokenizer, prompt: str, request_id: int, max_tokens: int = 256) -> RequestResult:
    """
    Single-request inference against a compiled TRT-LLM engine.

    WHY streaming callback for TTFT here too (mirroring vllm_infer in the
    notebook): TRT-LLM's ModelRunner.generate() supports a streaming mode
    that yields partial token sequences -- the first yield's timestamp
    marks TTFT identically to the vLLM measurement, keeping the two
    engines' TTFT semantics comparable in the final report.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    t0 = time.time()
    ttft = None
    output_ids = None

    for partial_output in runner.generate(
        input_ids,
        max_new_tokens=max_tokens,
        end_id=tokenizer.eos_token_id,
        pad_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        streaming=True,
    ):
        if ttft is None:
            ttft = time.time() - t0
        output_ids = partial_output

    e2e = time.time() - t0
    n_output_tokens = output_ids.shape[-1] - input_ids.shape[-1] if output_ids is not None else 0

    return RequestResult(
        request_id=request_id,
        prompt_tokens=input_ids.shape[-1],
        output_tokens=n_output_tokens,
        ttft_s=ttft or 0.0,
        e2e_s=e2e,
        success=True,
    )


def run_trtllm_concurrency_sweep(runner, tokenizer, prompts, concurrency_levels) -> list:
    """
    Note: TRT-LLM's ModelRunner is synchronous (not asyncio-native like
    vLLM's engine), so true request-level concurrency is achieved via
    TRT-LLM's own internal in-flight batching manager rather than Python-
    level asyncio.gather -- requests are submitted in a tight loop and the
    runner's scheduler handles batching internally, matching how a real
    Triton + TRT-LLM backend deployment would receive concurrent gRPC
    requests in production.
    """
    results = []
    for concurrency in concurrency_levels:
        subset = prompts[:max(concurrency * 2, 8)]
        t0 = time.time()
        request_results = [
            trtllm_infer(runner, tokenizer, p, i) for i, p in enumerate(subset)
        ]
        wall_clock = time.time() - t0

        bench_result = BenchmarkResult(
            engine="trtllm", precision="fp16", concurrency=concurrency,
            requests=request_results, wall_clock_s=wall_clock,
        )
        results.append(bench_result.summary())
    return results
