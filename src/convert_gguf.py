"""Convert the configured base HF model to GGUF, then quantize to the
target quant types (Q4_K_M, Q5_K_M, Q8_0 by default).

Runs inside envs/gguf/venv, which also clones llama.cpp for its
convert_hf_to_gguf.py script and the compiled `llama-quantize` binary.

Reads all hyperparameters from src/config.py's CONFIG singleton — see
that module for WHAT/WHY/EFFECT documentation on every value used here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONFIG

REPO_ROOT = Path(__file__).resolve().parent.parent
LLAMA_CPP_DIR = REPO_ROOT / "envs" / "gguf" / "llama.cpp"


def build_llama_quantize_binary():
    build_dir = LLAMA_CPP_DIR / "build"
    build_dir.mkdir(exist_ok=True)
    subprocess.run(["cmake", "-B", str(build_dir), "-DGGML_CUDA=ON"], cwd=LLAMA_CPP_DIR, check=True)
    subprocess.run(["cmake", "--build", str(build_dir), "--config", "Release", "-j"], cwd=LLAMA_CPP_DIR, check=True)
    binary = build_dir / "bin" / "llama-quantize"
    if not binary.exists():
        raise FileNotFoundError(f"Expected llama-quantize binary at {binary}")
    return binary


def main():
    model_id = CONFIG.model.hf_repo
    gguf_cfg = CONFIG.gguf
    output_dir = Path(gguf_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_model_dir = output_dir / "hf_source"
    snapshot_download(repo_id=model_id, local_dir=str(local_model_dir))

    fp16_gguf = output_dir / "model-f16.gguf"
    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    subprocess.run([
        "python3", str(convert_script), str(local_model_dir),
        "--outfile", str(fp16_gguf), "--outtype", "f16",
    ], check=True)

    quantize_binary = build_llama_quantize_binary()

    for quant_type in gguf_cfg.quant_types:
        out_path = output_dir / f"model-{quant_type}.gguf"
        subprocess.run([str(quantize_binary), str(fp16_gguf), str(out_path), quant_type], check=True)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
