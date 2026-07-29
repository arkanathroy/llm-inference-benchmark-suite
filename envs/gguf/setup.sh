#!/usr/bin/env bash
set -euo pipefail
ENV_DIR="envs/gguf/venv"
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install -q --upgrade pip
export CMAKE_ARGS="-DGGML_CUDA=on"
export FORCE_CMAKE=1
pip install -q llama-cpp-python --no-cache-dir
pip install -q huggingface_hub gguf sentencepiece protobuf numpy
pip install -q pynvml nvidia-ml-py pandas matplotlib plotly
pip install -q locust fastapi uvicorn httpx aiohttp pydantic python-dotenv rich tqdm
pip install -q lm-eval evaluate datasets scikit-learn transformers accelerate
if [ ! -d "envs/gguf/llama.cpp" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git envs/gguf/llama.cpp
fi
deactivate
echo "gguf environment ready at $ENV_DIR (llama.cpp cloned to envs/gguf/llama.cpp)"
