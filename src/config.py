"""
config.py
=========
Central configuration for the LLM Inference Optimization Benchmark Suite.

Every parameter below is documented with:
  WHAT   - what the parameter controls
  WHY    - why this specific value was chosen
  EFFECT - what happens if you raise / lower it

Model choice: Llama-3.2-3B-Instruct
  WHY 3B, not 7B or 70B: Colab free-tier T4 has 16GB VRAM. A 7B model in FP16
  already consumes ~14GB just for weights, leaving no headroom for KV cache
  or activation memory during batched inference. 3B in FP16 uses ~6GB, leaving
  ~10GB for KV cache experiments, quantization comparisons, and CUDA overhead.
  This mirrors the JD's benchmark scenarios (Whisper ASR, Llama 70B TTFT) at a
  scale actually reproducible on free infrastructure -- the methodology is
  identical, only the absolute numbers scale differently on A100/H100.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    # WHAT: HuggingFace repo ID for the base model.
    # WHY: 3B is the largest Llama model that fits comfortably in T4's 16GB
    #      across all four precision variants tested (FP32 baseline is
    #      skipped for exactly this reason -- 3B * 4 bytes = 12GB weights
    #      alone, too tight alongside activations).

    max_model_len: int = 4096
    # WHAT: Maximum context length (prompt + generation) the engine will
    #       allocate KV cache for.
    # WHY: 4096 covers realistic customer-support / RAG-style prompts (the
    #      benchmark's synthetic dataset uses 128-1024 token prompts) without
    #      over-allocating KV cache blocks that would never be used.
    # EFFECT: Doubling this to 8192 roughly doubles the KV cache memory
    #      reservation in vLLM (see gpu_memory_utilization below), which
    #      directly reduces the number of concurrent sequences you can serve.

    dtype: str = "float16"
    # WHY float16 over bfloat16: T4 (Turing architecture, compute capability
    # 7.5) has full-rate FP16 Tensor Core support but only emulated/slow
    # bfloat16. On A100/H100 (Ampere/Hopper) bfloat16 is preferred for its
    # wider dynamic range and is what you'd flip to in production.


@dataclass
class QuantizationConfig:
    """
    Every quantization method traded off along three axes:
      1. Compression ratio (VRAM saved)
      2. Throughput / latency change
      3. Accuracy delta on the eval benchmark

    WHY test 4 methods instead of 1: the JD explicitly asks for "rigorous
    quality-accuracy tradeoff analysis" across INT8, FP16, GPTQ, and AWQ --
    a single method proves you can run a script, four methods with a
    comparison table proves you understand the tradeoff space.
    """

    fp16_enabled: bool = True

    int8_enabled: bool = True
    int8_threshold: float = 6.0
    # WHAT: Outlier threshold for mixed-precision decomposition. Any
    #       activation value with magnitude > threshold is computed in FP16;
    #       everything else is quantized to INT8.
    # WHY 6.0: This is the value from the original LLM.int8() paper
    #       (Dettmers et al., 2022), empirically found to isolate the
    #       ~0.1% of "outlier features" that cause catastrophic accuracy
    #       loss if naively quantized. Values below 6.0 push more compute
    #       into the slower FP16 path (higher accuracy, lower speedup);
    #       above 6.0 quantizes more aggressively (faster, riskier).
    # EFFECT: Lowering to 4.0 increases the fraction of FP16 fallback
    #       computation, narrowing the speedup vs FP16 baseline but
    #       improving accuracy retention -- test this in the ablation cell.

    int8_skip_modules: tuple = ("lm_head",)
    # WHY: The final projection layer (lm_head) directly determines output
    #      token probabilities. Quantizing it causes disproportionate
    #      perplexity increase relative to the ~0.1% memory it represents.
    #      Standard practice across bitsandbytes, GPTQ, and AWQ alike.

    gptq_enabled: bool = True
    gptq_bits: int = 4
    # WHY 4-bit: The JD explicitly lists "4-bit" quantization as a
    #      requirement. 4-bit is also the sweet spot on the
    #      compression/accuracy Pareto frontier for 3-8B models -- 2-bit
    #      GPTQ shows severe degradation below ~7B parameters.

    gptq_group_size: int = 128
    # WHAT: Number of weight columns that share one quantization scale
    #       factor. Smaller groups = more scale factors stored = more
    #       memory overhead but finer-grained quantization = better accuracy.
    # WHY 128: The de facto standard from the GPTQ paper and used by nearly
    #       all published GPTQ checkpoints on HuggingFace. group_size=32
    #       gives marginally better accuracy at ~3% more memory overhead;
    #       group_size=-1 (per-channel) gives worst accuracy but smallest
    #       file size. 128 is the balanced default.
    # EFFECT: Halving to 64 typically recovers ~0.1-0.3 perplexity points
    #       at the cost of ~1.5x larger quantized weight files.

    gptq_damp_percent: float = 0.01
    # WHAT: Damping factor added to the Hessian diagonal during the
    #       Optimal Brain Quantization (OBQ) weight update solve, to
    #       prevent numerical instability when the Hessian is
    #       ill-conditioned (near-singular).
    # WHY 0.01: The GPTQ reference implementation default. Too low
    #       (e.g. 0.001) risks NaN/inf during the Cholesky decomposition
    #       on some layers; too high (e.g. 0.1) over-regularizes and
    #       degrades quantization accuracy.

    gptq_desc_act: bool = False
    # WHAT: "Activation order" -- whether to quantize weight columns in
    #       order of decreasing activation magnitude (True) rather than
    #       naive left-to-right order (False).
    # WHY False here: desc_act=True improves accuracy slightly but is
    #       NOT supported by vLLM's fused GPTQ kernels used later in the
    #       benchmark (Part 5) -- setting True would force the slow
    #       unfused kernel path and invalidate the throughput comparison.
    #       This is a real, commonly-hit compatibility gotcha documented
    #       in vLLM's GPTQ integration notes.

    gptq_calibration_samples: int = 128
    # WHY 128: GPTQ calibrates per-layer quantization scales using a
    #       forward pass over real text samples (here: a slice of
    #       WikiText-2). 128 samples is the standard used in the original
    #       paper -- fewer samples (32) show calibration variance across
    #       runs; more (512+) show diminishing returns for 2-4x the
    #       calibration wall-clock time.

    awq_enabled: bool = True
    awq_bits: int = 4
    awq_group_size: int = 128
    # Same rationale as GPTQ group_size above -- kept identical across
    # methods so the comparison table is apples-to-apples.

    awq_zero_point: bool = True
    # WHAT: Whether to use asymmetric quantization (zero_point=True, an
    #       independent zero-offset per group) vs symmetric (zero_point
    #       range is forced to be centered at 0).
    # WHY True: AWQ's core insight is that a small fraction of weight
    #       channels are salient (protected by scaling, not by precision).
    #       Asymmetric quantization better represents the actual weight
    #       distribution after AWQ's scale-search step, and is the
    #       setting used in every published AWQ checkpoint.

    awq_calib_dataset: str = "pileval"
    # WHY pileval over wikitext: AWQ's activation-aware scaling search
    #       specifically needs diverse activation statistics to find
    #       per-channel salience correctly. The Pile's validation split
    #       is the default in the reference AutoAWQ implementation because
    #       of its domain diversity (code, dialogue, technical text) vs
    #       WikiText's narrower encyclopedic register.


@dataclass
class VLLMConfig:
    """
    vLLM engine parameters. Documented individually because the JD lists
    PagedAttention, dynamic batching, and KV cache optimization as named,
    testable skills.
    """

    gpu_memory_utilization: float = 0.85
    # WHAT: Fraction of total GPU VRAM vLLM is allowed to reserve upfront
    #       for model weights + KV cache blocks + activation workspace.
    # WHY 0.85, not 0.9 (vLLM's own default): on a 16GB T4 shared with the
    #       Colab notebook kernel itself (which holds ~0.5-1GB for CUDA
    #       context, cudnn handles, etc.), leaving 0.9 reserved has caused
    #       observed OOM crashes when the benchmark's own client-side
    #       tensors (tokenizer buffers, logging arrays) compete for the
    #       remaining sliver. 0.85 leaves a safety margin validated
    #       empirically in this notebook's Part 3 stress test.
    # EFFECT: Raising this increases the number of KV cache blocks
    #       available, directly raising the max concurrent sequences
    #       vLLM can batch -- but too high causes CUDA OOM on shared/
    #       constrained GPUs like T4.

    block_size: int = 16
    # WHAT: Number of tokens' worth of KV cache stored per "page" in
    #       PagedAttention's block table (the OS-virtual-memory-inspired
    #       paging scheme vLLM uses to eliminate KV cache fragmentation).
    # WHY 16: vLLM's tested default across all supported GPU architectures.
    #       Smaller blocks (8) reduce internal fragmentation (less wasted
    #       KV cache space per sequence) but increase block-table lookup
    #       overhead per attention step. Larger blocks (32) reduce
    #       overhead but waste more memory when sequence lengths don't
    #       divide evenly into the block size. 16 is the empirically
    #       validated middle ground in vLLM's own benchmarks.

    max_num_seqs: int = 32
    # WHAT: Maximum number of sequences that can be batched together in a
    #       single decoding iteration (the ceiling on continuous batching
    #       width).
    # WHY 32: Chosen as the upper bound of the Locust concurrency sweep
    #       (Part 6 tests concurrency 1/4/8/16/32) -- setting max_num_seqs
    #       lower than your max tested concurrency would silently queue
    #       requests rather than truly batching them, invalidating the
    #       "throughput cliff" measurement the JD explicitly asks for.
    # EFFECT: This is a ceiling, not a target -- actual batch width at
    #       any instant is also bounded by available KV cache blocks
    #       (gpu_memory_utilization x total VRAM / block_size).

    max_num_batched_tokens: int = 8192
    # WHAT: Maximum total tokens (across all sequences) processed in one
    #       scheduler step, spanning both the prefill and decode phases.
    # WHY 8192 = 2x max_model_len: gives the scheduler room to batch a
    #       full-length prefill request alongside multiple in-flight
    #       decode-phase sequences (which only contribute 1 token each
    #       per step) without starving decode throughput. Setting this
    #       equal to max_model_len (4096) would let one long prefill
    #       monopolize an entire scheduler step, spiking TTFT for every
    #       other queued request -- this is precisely the "prefill vs
    #       decode contention" phenomenon the benchmark's concurrency
    #       sweep is designed to surface.

    enable_chunked_prefill: bool = True
    # WHAT: Splits long prompt prefill computation into smaller chunks
    #       interleaved with ongoing decode steps, instead of running
    #       prefill as one monolithic blocking step.
    # WHY True: Directly targets the TTFT-under-load problem above --
    #       without chunked prefill, a long incoming request can block
    #       token generation for every other active user for the full
    #       prefill duration. This is the mechanism vLLM uses to keep
    #       P99 TTFT bounded under mixed short/long-prompt traffic,
    #       which the benchmark's synthetic traffic mix (Part 6) is
    #       specifically designed to exercise.

    kv_cache_dtype: str = "auto"
    # WHY auto (defers to model dtype, FP16 here) rather than forcing
    #       fp8: FP8 KV cache roughly halves KV cache memory footprint
    #       but is only numerically validated on Ampere+ (A100/H100) in
    #       vLLM -- forcing it on T4 (Turing) is unsupported and will
    #       either error or silently fall back. The notebook's optional
    #       "Part 7: A100 extensions" cell demonstrates fp8 KV cache
    #       explicitly gated behind a GPU architecture check.

    swap_space_gb: int = 4
    # WHAT: CPU RAM (in GB) reserved for swapping out KV cache blocks of
    #       preempted sequences under memory pressure, rather than
    #       aborting them outright.
    # WHY 4: Colab's free-tier CPU RAM is ~12GB total; 4GB reserved for
    #       KV cache swap leaves sufficient headroom for the Python
    #       process, dataset loading, and OS overhead while still giving
    #       vLLM's scheduler a real recovery path during the high-
    #       concurrency (32) stress test instead of hard request failures.

    enforce_eager: bool = False
    # WHAT: Whether to disable CUDA graph capture and run every forward
    #       pass in eager PyTorch mode.
    # WHY False (i.e. CUDA graphs ARE used): CUDA graphs pre-record the
    #       sequence of GPU kernel launches for a given batch shape,
    #       eliminating Python/CPU dispatch overhead on replay -- this
    #       is a meaningful, measurable throughput contributor and
    #       disabling it would understate vLLM's real-world performance.
    #       The notebook includes an explicit ablation cell toggling this
    #       True to show CUDA graph capture's isolated contribution,
    #       since the JD asks for "bottleneck analysis."


@dataclass
class LocustConfig:
    """
    Load-testing configuration. The JD explicitly names Locust and asks
    for "performance cliffs, optimal concurrency, and scaling thresholds."
    """

    concurrency_levels: tuple = (1, 2, 4, 8, 16, 32)
    # WHY this specific sweep (not linear 1,10,20,30...): doubling at each
    #      step is standard practice for capacity testing because
    #      performance cliffs in batched serving systems are
    #      log-linear phenomena (queueing delay grows non-linearly past
    #      the saturation point) -- a linear sweep wastes samples in the
    #      uninteresting low-concurrency region and under-samples the
    #      critical inflection zone. This directly matches
    #      max_num_seqs=32 as the ceiling.

    spawn_rate: int = 4
    # WHAT: New simulated users started per second when ramping to a
    #       target concurrency level.
    # WHY 4: Fast enough that each concurrency plateau (Part 6) is
    #       reached within ~8 seconds even at the top-level 32 users,
    #       keeping total sweep runtime tractable on Colab's session
    #       limits, but slow enough to avoid a thundering-herd burst
    #       that would spike queueing latency from connection setup
    #       rather than from genuine model-serving saturation --
    #       conflating those two effects would corrupt the cliff
    #       measurement.

    run_time_per_level_s: int = 45
    # WHY 45s: Long enough to collect a statistically stable P50/P95/P99
    #       latency sample (at ~2-5 req/s per user this yields 100+
    #       requests per level) while keeping the full 6-level sweep
    #       under 5 minutes total -- well inside Colab's interactive
    #       session budget.

    prompt_length_mix: dict = field(default_factory=lambda: {
        "short": (32, 128, 0.4),
        "medium": (128, 512, 0.4),
        "long": (512, 1024, 0.2),
    })
    # WHY a mixed distribution instead of fixed-length prompts: real
    #      traffic (e.g. the JD's customer-support/voice-agent context)
    #      is never uniform length. A fixed-length benchmark hides the
    #      prefill/decode contention effect described in
    #      max_num_batched_tokens above. The 40/40/20 weighting
    #      approximates a typical chat-support prompt distribution
    #      (most turns short, some with pasted context, few very long).


@dataclass
class TRTLLMConfig:
    """
    TensorRT-LLM compilation parameters (Part 7, optional -- requires
    Colab Pro / A100 due to compilation memory and time requirements).
    """

    max_batch_size: int = 32
    max_input_len: int = 3072
    max_output_len: int = 1024
    precision: str = "float16"
    use_gpt_attention_plugin: str = "float16"
    use_paged_kv_cache: bool = True


@dataclass
class MonitoringConfig:
    dcgm_fields: tuple = (
        "DCGM_FI_DEV_GPU_UTIL",
        "DCGM_FI_DEV_FB_USED",
        "DCGM_FI_DEV_POWER_USAGE",
        "DCGM_FI_DEV_SM_CLOCK",
        "DCGM_FI_DEV_MEM_COPY_UTIL",
        "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
    )
    poll_interval_ms: int = 200


@dataclass
class CostConfig:
    instance_hourly_rates: dict = field(default_factory=lambda: {
        "g5.xlarge_A10G": 1.006,
        "g6.xlarge_L4": 0.805,
        "p4d.24xlarge_A100_40GB_x8": 32.77,
        "p5.48xlarge_H100_x8": 98.32,
        "colab_free_T4": 0.0,
        "colab_pro_A100": 1.18,
    })


CONFIG = {
    "model": ModelConfig(),
    "quant": QuantizationConfig(),
    "vllm": VLLMConfig(),
    "locust": LocustConfig(),
    "trtllm": TRTLLMConfig(),
    "monitor": MonitoringConfig(),
    "cost": CostConfig(),
}
