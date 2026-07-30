"""Shared helpers for starting/stopping backend servers (vLLM, llama.cpp)
and waiting for them to become healthy, with fail-fast behavior instead of
silently proceeding against a dead server. Used by every technique's
notebook cell so the startup/health-check logic lives in one place.
"""

from __future__ import annotations

import subprocess
import time

import httpx


def wait_for_health(base_url: str, proc: subprocess.Popen, max_wait_s: int = 300, poll_interval_s: int = 2):
    elapsed = 0
    while elapsed < max_wait_s:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Server process exited early with code {proc.returncode} "
                f"before becoming healthy. Check GPU memory / disk space."
            )
        try:
            if httpx.get(f"{base_url}/health", timeout=2).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(poll_interval_s)
        elapsed += poll_interval_s

    proc.terminate()
    raise RuntimeError(
        f"Server at {base_url} did not become healthy within {max_wait_s}s. "
        f"Aborting before running the benchmark against a dead server."
    )


def stop_server(proc: subprocess.Popen, timeout_s: int = 15):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout_s)


def start_vllm_server(python_bin, model_path, host, port, gpu_memory_utilization, max_model_len, quantization=None):
    cmd = [
        str(python_bin), "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_path),
        "--host", host,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
    ]
    if quantization:
        cmd += ["--quantization", quantization]
    return subprocess.Popen(cmd)


def start_llamacpp_server(server_binary, gguf_file, host, port, max_model_len):
    cmd = [
        str(server_binary),
        "-m", str(gguf_file),
        "--host", host,
        "--port", str(port),
        "-ngl", "-1",
        "-c", str(max_model_len),
    ]
    return subprocess.Popen(cmd)
