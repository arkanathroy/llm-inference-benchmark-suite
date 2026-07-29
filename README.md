# LLM Inference Benchmark Suite

A hands-on benchmarking project comparing **inference speed, memory footprint, and output quality**
across five ways of running the same open-weight LLM:

| # | Technique                | Library              | Precision            |
|---|---------------------------|-----------------------|------------------------|
| 1 | FP16 baseline              | Hugging Face Transformers | fp16                |
| 2 | GPTQ                       | GPTQModel              | 4-bit (INT4)          |
| 3 | AWQ                         | AutoAWQ                | 4-bit (INT4)          |
| 4 | GGUF                        | llama.cpp / llama-cpp-python | 4/5/8-bit (Q4_K_M, Q5_K_M, Q8_0) |
| 5 | TensorRT-LLM engine         | TensorRT-LLM            | fp16 / INT8 / INT4 (AWQ-in-TRT) |

Serving throughput for the HF/GPTQ/AWQ/GGUF paths is measured through **vLLM's** OpenAI-compatible
server (PagedAttention, continuous batching, chunked prefill), while the TensorRT-LLM path uses
its own native runtime, since vLLM does not execute compiled TRT engines directly.

## Why one virtualenv per technique

Earlier iterations of this project tried to install every quantization backend
(AutoGPTQ, AutoAWQ, Optimum, vLLM, TensorRT-LLM) into a **single environment**.
This consistently failed because these libraries pin **mutually incompatible
versions of `torch` and `transformers`**:

- `vllm` needs a torch build matching its compiled CUDA kernels (flash-attn / xformers).
- `GPTQModel` / `AutoAWQ` prebuilt CUDA kernels are compiled against a specific torch ABI.
- `TensorRT-LLM` ships its own pinned `torch` build tied to a specific CUDA Toolkit (13.x).
- `peft`/`accelerate`/`optimum` shift their own transformers floor and ceiling with every release.

Trying to satisfy all of these simultaneously in one resolver run is not solvable — the
constraints are genuinely contradictory, not just "pinned too tight." The fix used
throughout this repo is **one isolated Python virtual environment per technique**,
each with its own self-consistent, unpinned-to-latest-stable dependency set. A thin
orchestration layer (`src/env_runner.py`) invokes each venv's interpreter directly, so
the top-level orchestrator notebook never needs to resolve cross-technique
dependencies at all.

```
envs/
  fp16/         -> transformers + vllm only
  gptq/         -> GPTQModel + transformers + vllm
  awq/          -> autoawq + transformers + vllm
  gguf/         -> llama-cpp-python (built with CUDA) + huggingface_hub
  trtllm/       -> tensorrt_llm + its pinned torch/CUDA 13 stack
```

Each environment is built fresh by its own `envs/<name>/setup.sh` script, using
**latest stable releases** at build time (no hand-picked legacy pins), so the
project stays reproducible without freezing to any one point-in-time snapshot
that will eventually rot as upstream projects move forward.

## Repository layout

```
llm-inference-benchmark-suite/
├── README.md
├── configs/
│   ├── benchmark_config.yaml       # model id, prompts, batch sizes, output paths
│   └── sample_prompts.txt
├── envs/
│   ├── fp16/setup.sh
│   ├── gptq/setup.sh
│   ├── awq/setup.sh
│   ├── gguf/setup.sh
│   └── trtllm/setup.sh
├── src/
│   ├── gpu_monitor.py               # NVML-based GPU telemetry (util%, VRAM, power, temp)
│   ├── env_runner.py                 # subprocess bridge: run a script inside a given venv
│   ├── quantize_gptq.py              # GPTQModel quantization script (runs inside envs/gptq)
│   ├── quantize_awq.py               # AutoAWQ quantization script (runs inside envs/awq)
│   ├── convert_gguf.py               # HF -> GGUF conversion + llama.cpp quantize (envs/gguf)
│   ├── build_trtllm_engine.py        # TensorRT-LLM engine build script (envs/trtllm)
│   ├── vllm_server.py                 # start/stop a vLLM OpenAI-compatible server
│   ├── llamacpp_server.py             # start/stop a llama.cpp server for GGUF models
│   ├── trtllm_bench.py                # native TensorRT-LLM throughput/latency harness
│   └── benchmark_runner.py            # unified load generator + metrics collector (all backends)
├── notebooks/
│   └── llm_inference_benchmark_colab.ipynb   # orchestrator notebook (Colab, A100 recommended)
├── results/                          # CSV/JSON outputs + plots land here
└── docs/
    └── environment_notes.md          # troubleshooting notes specific to Colab GPU runtimes
```

## Hardware target

Built and tested against a **Google Colab A100 (40GB)** runtime. A T4 (16GB) runtime
also works for the FP16/GPTQ/AWQ/GGUF phases with a smaller model (<=3B parameters),
but the TensorRT-LLM phase requires an A100/H100-class GPU (Ampere or newer, SM80+)
since TensorRT-LLM's prebuilt kernels drop support for older architectures.

## Quickstart (Colab)

```python
!git clone https://github.com/arkanathroy/llm-inference-benchmark-suite.git
%cd llm-inference-benchmark-suite
!bash envs/fp16/setup.sh
!bash envs/gptq/setup.sh
!bash envs/awq/setup.sh
!bash envs/gguf/setup.sh
# TensorRT-LLM only if you have an A100/H100 runtime:
!bash envs/trtllm/setup.sh
```

Then open `notebooks/llm_inference_benchmark_colab.ipynb` and run the phases in order.
Each phase cell calls into its own venv via `src/env_runner.py`, so no single Python
process ever imports two conflicting quantization backends.

## What gets measured

For every technique, `benchmark_runner.py` records:

- **Time-to-first-token (TTFT)** and **inter-token latency**
- **Tokens/second** (single-stream and under concurrent load via `locust`)
- **Peak GPU memory** and **average GPU utilization** during the run (`gpu_monitor.py`)
- **Perplexity / task accuracy** delta vs the FP16 baseline (via `lm-eval`)
- **Model file size on disk** (checkpoint footprint)

Results are written as CSV to `results/` and rendered as comparison charts
(`results/*.png`) inside the notebook's final phase.

## Known limitations

- GGUF support in vLLM itself is still explicitly marked experimental upstream and only
  loads single-file GGUF checkpoints, so the GGUF phase benchmarks llama.cpp's native
  server as the primary path and vLLM-GGUF as a secondary, best-effort comparison.
- TensorRT-LLM's pip wheel pins its own `torch` build tied to a specific CUDA Toolkit
  version; mixing it into the same environment as vLLM/GPTQModel is not attempted —
  it always runs in its own isolated venv per the design above.
