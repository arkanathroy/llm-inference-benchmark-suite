#!/usr/bin/env bash
set -euo pipefail
ENV_DIR="envs/trtllm/venv"
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install -q --upgrade pip setuptools wheel
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu130
CURRENT_TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
echo "torch==${CURRENT_TORCH_VERSION}" > envs/trtllm/torch-constraint.txt
pip install -q --ignore-installed tensorrt_llm -c envs/trtllm/torch-constraint.txt
pip install -q huggingface_hub transformers accelerate
pip install -q pynvml nvidia-ml-py pandas matplotlib plotly
pip install -q locust fastapi uvicorn httpx aiohttp pydantic python-dotenv rich tqdm
deactivate
echo "trtllm environment ready at $ENV_DIR"
echo "NOTE: run notebook Phase 7 GPU-architecture check before using this env."
