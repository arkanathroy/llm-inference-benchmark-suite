"""Start/stop a vLLM OpenAI-compatible server as a background subprocess.

Used for the FP16, GPTQ, and AWQ phases (all served through vLLM's engine).
"""

from __future__ import annotations

import subprocess
import time
import urllib.request


class VLLMServer:
    def __init__(self, model_path, host="127.0.0.1", port=8000,
                 gpu_memory_utilization=0.90, max_model_len=4096,
                 quantization=None, extra_args=None):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.quantization = quantization
        self.extra_args = extra_args or []
        self.process = None

    def start(self, startup_timeout_s=180):
        cmd = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-model-len", str(self.max_model_len),
        ]
        if self.quantization:
            cmd += ["--quantization", self.quantization]
        cmd += self.extra_args

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
        raise TimeoutError(f"vLLM server did not become healthy within {timeout_s}s")

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
