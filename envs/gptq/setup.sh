#!/usr/bin/env bash
set -euo pipefail
ENV_DIR="envs/gptq/venv"
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install -q --upgrade pip
pip install -q vllm
pip install -q gptqmodel --no-build-isolation
pip install -q "transformers[sentencepiece]" accelerate
pip install -q pynvml nvidia-ml-py pandas numpy matplotlib plotly
pip install -q locust fastapi uvicorn httpx aiohttp pydantic python-dotenv rich tqdm
pip install -q lm-eval evaluate datasets scikit-learn
deactivate
echo "gptq environment ready at $ENV_DIR"
