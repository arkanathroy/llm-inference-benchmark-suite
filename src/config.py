"""Central configuration for the LLM Inference Benchmark Suite.

Every hyperparameter below is documented with WHAT it controls, WHY that
specific value was chosen, and the EFFECT of changing it — so a reader
never has to go spelunking through vLLM/GPTQModel/AutoAWQ/llama.cpp/
TensorRT-LLM source to understand a decision made here.

This module is the single source of truth for all five techniques
(FP16, GPTQ, AWQ, GGUF, TensorRT-LLM). Every script in src/ imports the
dataclasses below instead of parsing a YAML file directly, so IDE
autocomplete and type-checking catch typos in config keys before runtime.

Import pattern used by every other script in this repo:

    from config import CONFIG
    model_id = CONFIG.model.hf_repo
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # WHAT: Hugging Face repo id of the base model all five techniques quantize from.
    # WHY: Qwen2.5-3B-Instruct fits comfortably in a T4's 16GB VRAM at fp16
    #      (~6GB weights) with headroom for KV cache and activation memory,
    #      while still being large enough that quantization speedups are
    #      measurable (sub-1B models are often memory-bandwidth-bound
    #      regardless of precision, which would understate GPTQ/AWQ gains).
    # EFFECT: Swapping to a 7B+ model requires a T4 runtime upgrade to A100
    #      for the FP16 baseline phase, since 7B fp16 weights alone are ~14GB.
    hf_repo: str = "Qwen/Qwen2.5-3B-Instruct"

    # WHAT: Git revision/tag to pin the HF snapshot to.
    # WHY: "main" is convenient for iteration but not reproducible — model
    #      repos are occasionally updated in place. Pin to a specific commit
    #      hash before publishing final benchmark numbers.
    # EFFECT: Leaving this as "main" means re-running the pipeline weeks
    #      later could silently benchmark a different set of weights.
    revision: str = "main"

    # WHAT: Maximum context length (prompt + generation) the model/server
    #      will accept, passed as --max-model-len to vLLM and -c to
    #      llama.cpp's server.
    # WHY: 4096 covers the sample prompts (all under 200 tokens) plus
    #      max_new_tokens=256 generation with room to spare, without
    #      reserving KV cache for context the benchmark never uses.
    # EFFECT: Raising this increases the KV cache memory vLLM reserves
    #      per sequence, directly reducing gpu_memory_utilization headroom
    #      for concurrent requests — every doubling of max_model_len roughly
    #      halves the number of concurrent sequences that fit at a fixed
    #      gpu_memory_utilization ceiling.
    max_model_len: int = 4096


@dataclass
class GPTQConfig:
    # WHAT: Number of bits per weight in the quantized representation.
    # WHY: 4-bit is the standard GPTQ operating point — it's the precision
    #      the original GPTQ paper benchmarks against, and it's what
    #      GPTQModel's CUDA kernels are most optimized for. 3-bit exists
    #      but degrades quality noticeably faster than the size savings justify.
    # EFFECT: Dropping to 3-bit shrinks the checkpoint further but typically
    #      costs more perplexity than the AWQ 4-bit alternative, making the
    #      technique comparison less fair if only GPTQ goes to 3-bit.
    bits: int = 4

    # WHAT: Number of weights that share one scale/zero-point during
    #      quantization (GPTQ's group-wise quantization granularity).
    # WHY: 128 is the value used in the original GPTQ paper and is the
    #      de facto default across GPTQModel/AutoGPTQ downstream tooling.
    #      Smaller groups = finer-grained scales = better accuracy at the
    #      cost of extra scale/zero-point storage overhead.
    # EFFECT: group_size=32 recovers close to fp16 accuracy but adds ~4x
    #      more scale metadata than group_size=128, partially eating into
    #      the memory savings quantization is meant to deliver.
    group_size: int = 128

    # WHAT: Whether to reorder weight columns by decreasing activation
    #      magnitude before quantizing ("activation-order" / desc_act).
    # WHY: Disabled here because desc_act=True materially slows down
    #      GPTQ's CUDA kernel inference (column reordering breaks memory
    #      coalescing patterns the kernel relies on) for a quality gain
    #      that's marginal on a 3B-parameter model.
    # EFFECT: Setting this True would improve perplexity slightly at a
    #      real throughput cost — the opposite trade this benchmark suite
    #      is trying to measure fairly against AWQ (which has no equivalent
    #      knob), so it's kept off for an apples-to-apples comparison.
    desc_act: bool = False

    # WHAT: Directory the quantized checkpoint is written to.
    # EFFECT: benchmark_runner.py and vllm_server.py load the model
    #      straight from this path when serving the GPTQ variant.
    output_dir: str = "results/quantized/gptq"


@dataclass
class AWQConfig:
    # WHAT: Bit width, mirrors GPTQConfig.bits.
    # WHY: Kept identical to GPTQ's bit width (4-bit) so the two techniques
    #      are compared at matched compression ratios, not different ones.
    # EFFECT: Changing this without also changing GPTQConfig.bits breaks
    #      the apples-to-apples framing of the GPTQ vs AWQ comparison.
    bits: int = 4

    # WHAT: Group size for AWQ's per-group scaling, mirrors GPTQ's group_size.
    # WHY: Matched to GPTQConfig.group_size for the same fairness reason.
    # EFFECT: See GPTQConfig.group_size.
    group_size: int = 128

    # WHAT: Whether AWQ uses zero-point quantization (asymmetric) vs
    #      symmetric-only quantization.
    # WHY: True (asymmetric) is AutoAWQ's default and generally gives
    #      better accuracy for weight distributions that aren't
    #      zero-centered, which is common in practice.
    # EFFECT: Setting False forces symmetric quantization, which is
    #      slightly faster to dequantize but loses accuracy on
    #      non-symmetric weight distributions.
    zero_point: bool = True

    # WHAT: Directory the quantized checkpoint is written to.
    output_dir: str = "results/quantized/awq"


@dataclass
class GGUFConfig:
    # WHAT: List of llama.cpp quantization presets to produce from the
    #      fp16 GGUF conversion.
    # WHY: Q4_K_M, Q5_K_M, and Q8_0 span the three points GGUF users
    #      actually deploy in practice: Q4_K_M is the most common
    #      "good enough" default, Q5_K_M is the common quality/size
    #      compromise, and Q8_0 is the near-lossless upper bound used
    #      as a GGUF-side sanity check against the fp16 baseline.
    # EFFECT: Adding more entries here (e.g. "Q3_K_M", "Q6_K") linearly
    #      increases Phase 2.3's runtime, since each variant requires a
    #      separate llama-quantize invocation over the full fp16 GGUF file.
    quant_types: list = field(default_factory=lambda: ["Q4_K_M", "Q5_K_M", "Q8_0"])

    # WHAT: Directory the fp16 GGUF file and all quantized variants are
    #      written to.
    output_dir: str = "results/quantized/gguf"


@dataclass
class TRTLLMConfig:
    # WHAT: Base compute dtype passed to trtllm-build's --gemm_plugin flag.
    # WHY: float16 matches the FP16 baseline's precision, so the
    #      TensorRT-LLM phase without --use_awq measures pure kernel/engine
    #      compilation speedup, isolated from any quantization effect.
    # EFFECT: Switching to "bfloat16" only makes sense on GPUs with native
    #      bf16 Tensor Core support (Ampere+); on those GPUs it can reduce
    #      numerical overflow risk versus fp16 with no throughput cost.
    dtype: str = "float16"

    # WHAT: Whether to quantize the KV cache itself to INT8 in the compiled
    #      engine (separate from weight quantization).
    # WHY: Disabled by default because it's a TensorRT-LLM-only lever with
    #      no equivalent in vLLM/llama.cpp, so enabling it by default would
    #      make the cross-technique comparison uneven. Documented here as
    #      an opt-in extension for A100/H100 experiments.
    # EFFECT: Enabling this roughly halves KV cache memory, letting more
    #      concurrent sequences fit in the same VRAM budget, at a small
    #      (typically <1%) perplexity cost.
    int8_kv_cache: bool = False

    # WHAT: Whether to fold AWQ weight-only quantization directly into the
    #      TensorRT-LLM engine build (trtllm-build --quantization int4_awq).
    # WHY: True by default — this is what makes the TensorRT-LLM phase a
    #      genuinely distinct data point rather than a duplicate of the
    #      fp16 vLLM measurement: it answers "how much faster is AWQ
    #      inference through TensorRT-LLM's compiled engine versus vLLM's
    #      AWQ kernel path measured in Phase 5."
    # EFFECT: Setting False builds a full-precision (fp16) TensorRT-LLM
    #      engine instead, useful for isolating "TensorRT compilation
    #      speedup alone" from "TensorRT + AWQ combined speedup."
    use_awq: bool = True

    # WHAT: Directory the compiled TensorRT-LLM checkpoint/engine files
    #      are written to.
    output_dir: str = "results/quantized/trtllm_engine"


@dataclass
class BenchmarkConfig:
    # WHAT: Path to the newline-delimited prompt file used for all
    #      benchmark requests across every technique.
    prompts_file: str = "configs/sample_prompts.txt"

    # WHAT: Batch sizes swept during each technique's throughput benchmark.
    # WHY: 1 establishes single-stream latency (the TTFT/user-facing-latency
    #      number people usually care about first). 4/8/16/32 span the
    #      range where vLLM's continuous batching and PagedAttention start
    #      to show throughput gains over naive batching — this is the
    #      regime the whole "continuous batching" pitch is about, so the
    #      sweep needs to cover it, not just batch_size=1.
    # EFFECT: batch_size values above ~32 risk exceeding a T4's KV cache
    #      capacity at max_model_len=4096 (see ModelConfig.max_model_len docs);
    #      raising this ceiling requires either shrinking max_model_len or
    #      moving to an A100.
    batch_sizes: list = field(default_factory=lambda: [1, 4, 8, 16, 32])

    # WHAT: Concurrent virtual user counts used by the Locust-based load
    #      test variant of the benchmark (see benchmark_runner.py).
    # WHY: Chosen to bracket the batch_sizes sweep — 20 concurrent users is
    #      intentionally below the batch_size=32 ceiling above, since
    #      Locust's arrival pattern is bursty (not perfectly batched),
    #      so its effective concurrent-request peak needs headroom
    #      below the hard batching ceiling to avoid request queuing
    #      artifacts contaminating the latency measurements.
    concurrent_users: list = field(default_factory=lambda: [1, 5, 10, 20])

    # WHAT: Number of tokens generated per benchmark request.
    # WHY: 256 is long enough to move past the prefill-dominated startup
    #      phase into steady-state decode (where inter-token latency
    #      differences between techniques actually show up), while short
    #      enough to keep a full 5-technique x 5-batch-size sweep within a
    #      Colab session's practical time budget.
    # EFFECT: Increasing this shifts total_latency further toward
    #      decode-bound time, making tokens/second comparisons more
    #      representative of steady-state throughput and less sensitive
    #      to prefill/TTFT overhead differences between techniques.
    max_new_tokens: int = 256

    # WHAT: Number of throwaway requests fired before timing starts.
    # WHY: 3 is enough to trigger vLLM's/TensorRT-LLM's JIT kernel
    #      autotuning and CUDA graph capture (both of which add one-time
    #      latency to the very first few requests after server startup),
    #      so those costs don't leak into the measured average.
    # EFFECT: Setting this to 0 will show artificially high latency /
    #      low tokens/sec on the first measured request of every batch
    #      size, skewing the average — especially noticeable at
    #      batch_size=1 where there's only ~20 samples to average over.
    num_warmup_requests: int = 3

    # WHAT: Number of timed requests averaged per batch size.
    # WHY: 20 balances statistical stability (reduces the influence of any
    #      single slow/fast outlier request) against total notebook runtime
    #      across a 5-batch-size x 5-technique sweep.
    # EFFECT: Dropping this below ~10 makes the reported avg_tokens_per_second
    #      noticeably noisy between re-runs of the same technique/batch_size,
    #      since GPU clock throttling and background OS jitter aren't
    #      averaged out.
    num_measured_requests: int = 20


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"

    # WHAT: Port vLLM's OpenAI-compatible server listens on, used for the
    #      FP16, GPTQ, and AWQ phases.
    vllm_port: int = 8000

    # WHAT: Port llama.cpp's server listens on, used for the GGUF phase.
    # WHY: Different from vllm_port so both servers could theoretically
    #      run side by side for a direct head-to-head request if needed,
    #      without a port collision.
    llamacpp_port: int = 8001

    # WHAT: Fraction of total GPU memory vLLM is allowed to pre-allocate
    #      for weights + KV cache + activations.
    # WHY: 0.90 rather than leaving vLLM's own 0.90 default implicit —
    #      pinned explicitly here so the value is visible and adjustable
    #      in one place. On a T4 (16GB) this leaves ~1.6GB free for the
    #      CUDA context, NCCL buffers, and OS/driver overhead, which in
    #      practice is enough margin to avoid OOM at the 3B model size
    #      configured above; on models close to a T4's ceiling, 0.90 can
    #      OOM and needs to drop to 0.85.
    # EFFECT: Raising this toward 1.0 risks CUDA OOM crashes that abort
    #      the whole benchmark phase; lowering it reduces the max KV cache
    #      size, which lowers the maximum sustainable batch_size/concurrent_users
    #      before requests start queuing.
    gpu_memory_utilization: float = 0.90


@dataclass
class EvalConfig:
    # WHAT: lm-eval task names run against each technique's server to
    #      measure accuracy delta from quantization.
    # WHY: hellaswag (commonsense sentence completion) and arc_easy
    #      (grade-school science QA) are both fast multiple-choice tasks
    #      lm-eval can score without needing free-form generation, keeping
    #      per-technique eval time low relative to the throughput
    #      benchmarking phases.
    tasks: list = field(default_factory=lambda: ["hellaswag", "arc_easy"])

    # WHAT: Number of eval samples drawn per task.
    # WHY: 200 gives a stable-enough accuracy estimate (roughly +/-3-4
    #      percentage points of noise at this sample size for a
    #      binary/multiple-choice task) while keeping each technique's
    #      eval pass to a few minutes on a T4, rather than running the
    #      full task (often 10k+ samples).
    # EFFECT: Lowering this toward ~50 makes the reported accuracy delta
    #      between fp16 and a quantized variant too noisy to distinguish
    #      from sampling variance, defeating the point of the comparison.
    limit: int = 200


@dataclass
class OutputConfig:
    results_dir: str = "results"
    csv_name: str = "benchmark_results.csv"
    chart_name: str = "benchmark_comparison.png"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    gptq: GPTQConfig = field(default_factory=GPTQConfig)
    awq: AWQConfig = field(default_factory=AWQConfig)
    gguf: GGUFConfig = field(default_factory=GGUFConfig)
    trtllm: TRTLLMConfig = field(default_factory=TRTLLMConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# Singleton instance every script imports. Override individual fields at
# the call site if you need a one-off experiment, e.g.:
#     from config import CONFIG
#     CONFIG.benchmark.max_new_tokens = 512
CONFIG = Config()


if __name__ == "__main__":
    from dataclasses import asdict
    import json
    print(json.dumps(asdict(CONFIG), indent=2))
