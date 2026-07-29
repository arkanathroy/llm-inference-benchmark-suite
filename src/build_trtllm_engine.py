"""Build a TensorRT-LLM engine from the configured base model.

Runs inside envs/trtllm/venv. Requires an Ampere-or-newer GPU (A100/H100,
SM80+); raises a clear assertion error rather than silently failing
partway through compilation if run on an unsupported architecture (e.g.
Colab's free-tier T4, which is Turing/SM75).

Reads all hyperparameters from src/config.py's CONFIG singleton — see
that module for WHAT/WHY/EFFECT documentation on every value used here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONFIG

MIN_SUPPORTED_SM_MAJOR = 8


def assert_gpu_supported():
    assert torch.cuda.is_available(), "No CUDA GPU detected."
    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    assert major >= MIN_SUPPORTED_SM_MAJOR, (
        f"TensorRT-LLM engine build requires an Ampere-or-newer GPU (SM80+). "
        f"Detected '{name}' with compute capability sm_{major}{minor}, which is "
        f"below the minimum. Switch this Colab runtime to an A100/H100 GPU."
    )
    print(f"GPU check passed: {name} (sm_{major}{minor})")


def main():
    assert_gpu_supported()

    model_id = CONFIG.model.hf_repo
    trt_cfg = CONFIG.trtllm
    output_dir = Path(trt_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_model_dir = output_dir / "hf_source"
    snapshot_download(repo_id=model_id, local_dir=str(local_model_dir))

    engine_dir = output_dir / "trt_engine"

    build_cmd = [
        "trtllm-build",
        "--checkpoint_dir", str(local_model_dir),
        "--output_dir", str(engine_dir),
        "--gemm_plugin", trt_cfg.dtype,
    ]
    if trt_cfg.use_awq:
        build_cmd += ["--quantization", "int4_awq"]
    elif trt_cfg.int8_kv_cache:
        build_cmd += ["--kv_cache_type", "int8"]

    subprocess.run(build_cmd, check=True)
    print(f"TensorRT-LLM engine written to {engine_dir}")


if __name__ == "__main__":
    main()
