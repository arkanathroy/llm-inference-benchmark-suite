# LLM Inference Optimization Benchmark Suite

Project 3 of a 4-project GPU/ML Systems Engineer portfolio, built to
demonstrate hands-on proficiency with the exact tech stack named in a
target GPU/ML Systems Engineer job description: **vLLM, TensorRT-LLM,
model quantization (INT8/FP16/GPTQ/AWQ), Locust load testing, GPU
profiling (DCGM-equivalent), and cost-per-token benchmark reporting.**

## What This Project Demonstrates

A systematic, end-to-end benchmark of **Llama-3.2-3B-Instruct** across
every layer of the LLM inference optimization stack:

1. **Quantization tradeoffs** — FP16 baseline vs INT8 (bitsandbytes) vs
   GPTQ (4-bit) vs AWQ (4-bit), measured on memory footprint, raw
   inference speed, perplexity, and MMLU task accuracy.
2. **Serving engine gains** — vLLM's PagedAttention, continuous
   batching, and chunked prefill layered on top of each precision
   variant.
3. **Load testing under realistic concurrency** — a Locust-driven
   sweep (1 → 32 concurrent users) that exposes the exact concurrency
   level where latency "cliffs" — the point where the serving engine's
   KV cache capacity saturates and requests start queueing.
4. **GPU telemetry** — utilization, VRAM, power, and clock speed
   captured during the load test, using a `pynvml`-based collector that
   maps 1:1 onto real DCGM field names (since Colab has no DCGM daemon
   available).
5. **Cost-per-token reporting** — measured throughput translated into
   USD-per-million-tokens across 6 real AWS/Colab instance price points,
   producing an actionable "which instance should I deploy on" table.
6. **(Optional) TensorRT-LLM compilation** — gated behind a GPU
   architecture check, runnable on Colab Pro/A100 to compare compiled-
   engine throughput against vLLM.

## Why Llama-3.2-3B, Not a Larger Model

Colab's free-tier T4 GPU has 16GB of VRAM. A 7B model in FP16 alone
consumes ~14GB, leaving no room for KV cache or activation memory
during concurrent request testing. **3B in FP16 uses ~6GB**, leaving
roughly 10GB of headroom to run four precision variants, a vLLM engine
with KV cache, and a 32-way concurrency sweep — all within free
infrastructure.

The methodology (not the absolute numbers) is what transfers to
production scale: the same benchmark structure run against a 70B model
on 8xA100/H100 would surface the same categories of bottlenecks
(prefill/decode contention, KV cache saturation, quantization accuracy
tradeoffs), just at different absolute latency/throughput values. This
mirrors the reference job posting's own benchmark examples ("0.41s
TTFT on Llama 70B") — the skill being demonstrated is the benchmarking
*methodology*, reproducible on any GPU tier.

## Repository Structure

```
llm-inference-benchmark-suite/
├── README.md
├── requirements.txt
├── notebooks/
│   └── llm_inference_benchmark_colab.ipynb   <- Main deliverable, run top-to-bottom on Colab T4
├── src/
│   ├── config.py              <- Every hyperparameter, documented with WHAT/WHY/EFFECT
│   ├── quantize.py             <- FP16 / INT8 / GPTQ / AWQ load + quantize functions
│   ├── evaluate.py             <- Perplexity (WikiText-2) + MMLU accuracy evaluation
│   ├── benchmark_runner.py     <- TTFT/TPOT/throughput measurement harness (async)
│   ├── prompt_generator.py     <- Synthetic prompt generator (controlled token-length mix)
│   ├── vllm_server.py          <- vLLM AsyncLLMEngine + OpenAI-server launcher
│   ├── trtllm_bench.py         <- TensorRT-LLM benchmark harness (Colab Pro/A100 only)
│   ├── gpu_monitor.py          <- pynvml-based DCGM-equivalent telemetry collector
│   └── plotting.py             <- Plotly chart generation for the final report
├── configs/
│   └── locustfile.py           <- Locust load test definition (targets vLLM's OpenAI API)
└── benchmarks/                 <- Output directory: CSVs, PNGs, BENCHMARK_REPORT.md
```

## Quickstart (Google Colab)

1. Open `notebooks/llm_inference_benchmark_colab.ipynb` in Colab.
2. **Runtime > Change runtime type > T4 GPU** (free tier is sufficient
   for Phases 1-8).
3. Add your HuggingFace token to Colab Secrets as `HF_TOKEN` (key icon,
   left sidebar) — Llama-3.2 is a gated model requiring license
   acceptance at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct.
4. Run cells top to bottom. Phase 2 (GPTQ/AWQ calibration) takes
   5-10 minutes each on first run; results are cached to Google Drive
   so subsequent runs skip re-calibration.
5. Phase 9 (TensorRT-LLM) requires switching to a Colab Pro A100
   runtime — the notebook raises a clear assertion error if run on T4
   rather than silently failing partway through compilation.

## Key Parameter Decisions (Summary — full rationale in `src/config.py`)

| Parameter | Value | One-line rationale |
|---|---|---|
| `gpu_memory_utilization` | 0.85 | Leaves safety margin below vLLM's 0.9 default on a T4 shared with notebook kernel overhead |
| `block_size` (PagedAttention) | 16 | vLLM's empirically validated default across GPU architectures |
| `max_num_seqs` | 32 | Matched to the top of the Locust concurrency sweep so batching isn't artificially capped below the tested load |
| `max_num_batched_tokens` | 8192 | 2x max_model_len — gives the scheduler room to interleave one long prefill with ongoing decodes without starving them |
| `enable_chunked_prefill` | True | Prevents one long prompt from blocking TTFT for every other concurrent user |
| GPTQ `group_size` | 128 | Standard from the original paper; balances accuracy vs scale-factor memory overhead |
| GPTQ `desc_act` | False | Required False for compatibility with vLLM's fused GPTQ kernel — a real, documented gotcha |
| AWQ `zero_point` | True | Matches every published AWQ checkpoint; correctly represents asymmetric weight distributions post-scaling |
| Locust `concurrency_levels` | (1,2,4,8,16,32) | Doubling sweep — performance cliffs in batched serving are log-linear phenomena, not linear |
| INT8 `llm_int8_threshold` | 6.0 | Original LLM.int8() paper's empirically found outlier-isolation threshold |

## GPU Metrics: Colab pynvml vs Production DCGM

Colab's sandboxed runtime has no root/systemd access, so the real DCGM
daemon (`dcgm-exporter`) cannot run. `src/gpu_monitor.py` polls the same
underlying NVIDIA driver counters via `pynvml` at 200ms resolution and
documents an explicit field-name mapping so the results translate
directly to a production Prometheus + DCGM Exporter setup:

| This notebook's metric | Real DCGM field ID |
|---|---|
| `gpu_util_pct` | `DCGM_FI_DEV_GPU_UTIL` |
| `mem_used_mb` | `DCGM_FI_DEV_FB_USED` |
| `power_watts` | `DCGM_FI_DEV_POWER_USAGE` |
| `sm_clock_mhz` | `DCGM_FI_DEV_SM_CLOCK` |
| `mem_copy_util_pct` | `DCGM_FI_DEV_MEM_COPY_UTIL` |

Tensor Core utilization (`DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`) is not
exposed via `pynvml` on consumer/T4 driver builds — this requires DCGM
profiling permissions typically only available on A100/H100 in a
production Kubernetes/EKS deployment, and is noted explicitly in the
monitor's summary output rather than silently omitted.

## Results (Populated After Running the Notebook)

The notebook writes all outputs to
`/content/drive/MyDrive/llm-inference-benchmark-suite/benchmarks/`,
including:

- `quantization_tradeoff_table.csv` — memory / speed / perplexity / MMLU accuracy per precision
- `combined_vllm_concurrency.csv` — TTFT/TPOT/throughput at every concurrency level, every precision
- `locust_sweep_summary.csv` — real Locust-measured P50/P95/P99 TTFT and RPS
- `gpu_telemetry_c32.csv` — raw GPU utilization/memory/power time series during peak load
- `cost_per_million_tokens.csv` — cost comparison across 6 instance types
- `BENCHMARK_REPORT.md` — consolidated markdown report combining all of the above

## Updating Instance Pricing

`src/config.py::CostConfig.instance_hourly_rates` hardcodes AWS
on-demand pricing (us-east-1) as of the notebook's last run. AWS
pricing changes periodically — update these values from the [AWS EC2
pricing page](https://aws.amazon.com/ec2/pricing/on-demand/) before
treating the cost report as current for a real deployment decision.

## Extending This Project

- **Swap the model**: change `ModelConfig.model_id` in `src/config.py`
  to any HuggingFace causal LM. Re-run Phase 2's calibration cells for
  the new model.
- **Add FP8 KV cache** (A100/H100 only): set
  `VLLMConfig.kv_cache_dtype = "fp8"` inside a GPU-architecture-gated
  cell, mirroring the Phase 9 TensorRT-LLM guard pattern.
- **Add speaker/task-specific eval sets**: `src/evaluate.py`'s
  `evaluate_mmlu_subset()` accepts any MMLU `subject` string — swap in
  a domain-relevant subject if adapting this for a specific production
  use case (e.g. `professional_medicine` for a healthcare LLM).
