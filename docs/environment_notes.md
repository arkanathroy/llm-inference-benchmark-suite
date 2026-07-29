# Colab environment notes

## Why one venv per technique

See the top-level README for the full rationale. In short: `vllm`,
`gptqmodel`, `autoawq`, `llama-cpp-python`, and `tensorrt_llm` each pin or
compile against a specific `torch` build, and those pins are not
mutually satisfiable in a single environment. Each `envs/<name>/setup.sh`
builds a fresh `venv` so pip only ever has to solve one technique's
constraints at a time.

## GPU architecture requirements

| Phase   | Minimum GPU                     | Notes                                            |
|---------|----------------------------------|---------------------------------------------------|
| FP16    | Any CUDA GPU (T4 works)          | Model must fit in VRAM at fp16.                    |
| GPTQ    | Any CUDA GPU (T4 works)          | GPTQModel kernels support Turing (SM75) and newer. |
| AWQ     | Any CUDA GPU (T4 works)          | AutoAWQ kernels support Turing (SM75) and newer.   |
| GGUF    | Any CUDA GPU, or CPU-only        | llama.cpp runs on CPU too; GPU offload is optional.|
| TensorRT-LLM | Ampere or newer (A100/H100, SM80+) | `build_trtllm_engine.py` asserts this and fails fast with a clear message on T4. |

## Known rough edges

- **condacolab checksum drift**: if bootstrapping via `condacolab`, pin to a
  specific installer URL — Anaconda occasionally re-uploads installer
  artifacts under the same version tag.
- **Colab Python version drift**: the base Colab runtime's Python version
  changes across image updates. All `envs/*/setup.sh` scripts use
  `python3 -m venv`, which inherits whatever Python is already active in
  the Colab runtime, so there's no assumption baked in about an exact
  minor version.
- **vLLM + xformers/flash-attn**: don't pin `torch` yourself before
  installing `vllm` in the fp16/gptq/awq envs. Let `pip install vllm`
  resolve its own torch version first; installing GPTQModel/AutoAWQ
  afterward (which compile against whatever torch is already present)
  avoids the ABI mismatch that caused "CUDA extension not installed"
  fallbacks in earlier iterations of this project.
- **TensorRT-LLM torch pin**: always regenerate `torch-constraint.txt` from
  the actually-installed torch version before installing `tensorrt_llm`,
  rather than hardcoding a version number, since the pip wheel's expected
  torch build changes across TensorRT-LLM releases.
