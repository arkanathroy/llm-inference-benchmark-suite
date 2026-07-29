"""GPU telemetry collector using NVML (pynvml / nvidia-ml-py).

Runs as a background sampling thread so a benchmark script can start it,
run its workload, then stop it and pull summary statistics (peak VRAM,
average utilization, average power draw).

Note on Tensor Core utilization: pynvml only exposes coarse-grained
utilization (SM busy %, memory busy %) via nvmlDeviceGetUtilizationRates.
Fine-grained Tensor Core pipe activity (DCGM_FI_PROF_PIPE_TENSOR_ACTIVE)
requires DCGM profiling counters, which need elevated permissions typically
only available on dedicated A100/H100 instances, not shared T4 Colab
runtimes. This module reports what NVML exposes on whatever GPU is present
and clearly labels Tensor Core metrics as unavailable when DCGM access
fails, rather than silently omitting them.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import pynvml
except ImportError as exc:
    raise ImportError(
        "pynvml is required. Install with `pip install pynvml nvidia-ml-py`."
    ) from exc


@dataclass
class GPUSample:
    timestamp: float
    util_percent: float
    mem_used_mb: float
    mem_total_mb: float
    power_watts: float
    temperature_c: float


@dataclass
class GPUMonitor:
    device_index: int = 0
    sample_interval_s: float = 0.5
    samples: list = field(default_factory=list)
    _thread: object = field(default=None, init=False, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _handle: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            except pynvml.NVMLError:
                power = float("nan")
            try:
                temp = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
            except pynvml.NVMLError:
                temp = float("nan")

            self.samples.append(GPUSample(
                timestamp=time.time(),
                util_percent=float(util.gpu),
                mem_used_mb=mem.used / (1024 ** 2),
                mem_total_mb=mem.total / (1024 ** 2),
                power_watts=power,
                temperature_c=temp,
            ))
            self._stop_event.wait(self.sample_interval_s)

    def start(self) -> None:
        self.samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return self.summary()

    def summary(self) -> dict:
        if not self.samples:
            return {
                "peak_mem_used_mb": None,
                "avg_util_percent": None,
                "avg_power_watts": None,
                "avg_temperature_c": None,
                "num_samples": 0,
                "tensor_core_active_percent": "unavailable (requires DCGM profiling permissions, not exposed by pynvml on most Colab GPUs)",
            }
        power_samples = [s.power_watts for s in self.samples if s.power_watts == s.power_watts]
        temp_samples = [s.temperature_c for s in self.samples if s.temperature_c == s.temperature_c]
        return {
            "peak_mem_used_mb": max(s.mem_used_mb for s in self.samples),
            "avg_util_percent": sum(s.util_percent for s in self.samples) / len(self.samples),
            "avg_power_watts": sum(power_samples) / max(1, len(power_samples)),
            "avg_temperature_c": sum(temp_samples) / max(1, len(temp_samples)),
            "num_samples": len(self.samples),
            "tensor_core_active_percent": "unavailable (requires DCGM profiling permissions, not exposed by pynvml on most Colab GPUs)",
        }

    def save_raw(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([s.__dict__ for s in self.samples], f, indent=2)


if __name__ == "__main__":
    mon = GPUMonitor()
    mon.start()
    time.sleep(5)
    print(json.dumps(mon.stop(), indent=2))
