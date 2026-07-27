"""
vllm_server.py
==============
Programmatic launcher for the vLLM engine, used both for the notebook's
in-process AsyncLLMEngine benchmarking (Phase 4) and for spawning the
OpenAI-compatible HTTP server that Locust targets (Phase 6).

Two distinct usage modes, deliberately kept separate:

1. AsyncLLMEngine (in-process): used for the controlled, fine-grained
   TTFT/TPOT benchmark in Phase 4 -- avoids HTTP/serialization overhead
   so the measured numbers reflect the ENGINE's performance, not the
   web server framework's overhead.

2. OpenAI-compatible server subprocess: used for the Locust load test
   in Phase 6 -- because Locust is an HTTP load generator, this mode
   deliberately INCLUDES realistic HTTP overhead, matching how the
   system would actually be consumed by real client applications.
"""

import subprocess
import time
import httpx

from config import CONFIG

VLLM_CFG = CONFIG["vllm"]
MODEL_CFG = CONFIG["model"]


def build_engine_args(quantization: str = None) -> dict:
    """
    Assembles vLLM EngineArgs kwargs. quantization=None loads the
    unquantized FP16 checkpoint; "gptq" or "awq" load the pre-quantized
    checkpoints saved by quantize.py, with vLLM auto-detecting and using
    its own fused quantized-matmul CUDA kernels for that format -- this
    is a DIFFERENT (faster) code path than the transformers-library
    dequantization used when just calling model.generate() directly in
    Phase 2, which is why the same quantization method shows different
    throughput numbers in Phase 2 (raw HF) vs Phase 5 (vLLM-served).
    """
    args = dict(
        model=MODEL_CFG.model_id,
        dtype=MODEL_CFG.dtype,
        max_model_len=MODEL_CFG.max_model_len,
        gpu_memory_utilization=VLLM_CFG.gpu_memory_utilization,
        block_size=VLLM_CFG.block_size,
        max_num_seqs=VLLM_CFG.max_num_seqs,
        max_num_batched_tokens=VLLM_CFG.max_num_batched_tokens,
        enable_chunked_prefill=VLLM_CFG.enable_chunked_prefill,
        kv_cache_dtype=VLLM_CFG.kv_cache_dtype,
        swap_space=VLLM_CFG.swap_space_gb,
        enforce_eager=VLLM_CFG.enforce_eager,
    )
    if quantization:
        args["quantization"] = quantization
        args["model"] = f"./{quantization}_model"
    return args


def launch_openai_server(quantization: str = None, port: int = 8000) -> subprocess.Popen:
    """
    Launches `vllm serve` as a background subprocess exposing the
    OpenAI-compatible /v1/chat/completions endpoint that locustfile.py
    targets.

    WHY subprocess rather than importing vllm.entrypoints.openai in-
    process: the OpenAI server's own internal event loop and the
    notebook kernel's event loop would otherwise conflict -- launching
    as an isolated subprocess is vLLM's own officially documented
    pattern for notebook/Colab environments.
    """
    args = build_engine_args(quantization)
    cmd = ["python", "-m", "vllm.entrypoints.openai.api_server", "--port", str(port)]
    for k, v in args.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd += [flag, str(v)]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    _wait_for_server_ready(port)
    return proc


def _wait_for_server_ready(port: int, timeout_s: int = 180):
    """
    Polls the health endpoint rather than sleeping a fixed duration,
    because vLLM's startup time varies significantly by quantization
    method -- GPTQ/AWQ checkpoint loading + CUDA kernel warmup can take
    2-3x longer than loading the plain FP16 checkpoint.
    """
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = httpx.get(f"http://localhost:{port}/health", timeout=2.0)
            if r.status_code == 200:
                return
        except httpx.RequestError:
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM server did not become ready within {timeout_s}s")
