"""Start/stop llama.cpp's built-in OpenAI-compatible server for a GGUF model.

Runs from within envs/gguf/venv. Uses the `llama-server` binary built
alongside `llama-quantize` in src/convert_gguf.py's CMake build step.
"""

from __future__ import annotations

import subprocess
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LLAMA_SERVER_BINARY = REPO_ROOT / "envs" / "gguf" / "llama.cpp" / "build" / "bin" / "llama-server"


class LlamaCppServer:
    def __init__(self, gguf_path, host="127.0.0.1", port=8001, n_gpu_layers=-1, ctx_size=4096):
        self.gguf_path = gguf_path
        self.host = host
        self.port = port
        self.n_gpu_layers = n_gpu_layers
        self.ctx_size = ctx_size
        self.process = None

    def start(self, startup_timeout_s=120):
        if not LLAMA_SERVER_BINARY.exists():
            raise FileNotFoundError(
                f"llama-server binary not found at {LLAMA_SERVER_BINARY}. "
                "Run src/convert_gguf.py's build step first."
            )
        cmd = [
            str(LLAMA_SERVER_BINARY), "-m", self.gguf_path,
            "--host", self.host, "--port", str(self.port),
            "-ngl", str(self.n_gpu_layers), "-c", str(self.ctx_size),
        ]
        self.process = subprocess.Popen(cmd)
        self._wait_for_health(startup_timeout_s)

    def _wait_for_health(self, timeout_s):
        url = f"http://{self.host}:{self.port}/health"
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                pass
            time.sleep(2)
        raise TimeoutError(f"llama.cpp server did not become healthy within {timeout_s}s")

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    @property
    def base_url(self):
        return f"http://{self.host}:{self.port}/v1"
