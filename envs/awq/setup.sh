#!/usr/bin/env bash
set -euo pipefail
ENV_DIR="envs/awq/venv"
if [ ! -d "$ENV_DIR" ] || [ ! -x "$ENV_DIR/bin/python3" ]; then
  rm -rf "$ENV_DIR"
  if python3 -m venv "$ENV_DIR" 2>/tmp/venv_err.log; then
    echo "venv created with standard ensurepip bootstrap"
  else
    echo "Standard venv creation failed (broken ensurepip), falling back to --without-pip + get-pip.py"
    cat /tmp/venv_err.log
    python3 -m venv "$ENV_DIR" --without-pip
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    "$ENV_DIR/bin/python3" /tmp/get-pip.py
  fi
else
  echo "Reusing existing venv at $ENV_DIR"
fi
source "$ENV_DIR/bin/activate"
pip install -q --upgrade pip
pip install -q vllm
pip install -q llmcompressor
pip install -q "transformers[sentencepiece]" accelerate
pip install -q pynvml nvidia-ml-py pandas numpy matplotlib plotly
pip install -q locust fastapi uvicorn httpx aiohttp pydantic python-dotenv rich tqdm
pip install -q lm-eval evaluate datasets scikit-learn
deactivate
echo "awq environment ready at $ENV_DIR"
