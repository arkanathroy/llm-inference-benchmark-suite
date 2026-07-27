"""
gpu_monitor.py
==============
DCGM-equivalent GPU telemetry collector for environments without a DCGM
daemon (e.g. Colab, which has no root/systemd access to run dcgm-exporter).

WHY pynvml instead of shelling out to `nvidia-smi dmon`:
  nvidia-smi dmon's text output requires fragile regex parsing and its
  polling interval floors around 1s. pynvml (NVIDIA Management Library
  Python bindings) gives structured, typed access to the same underlying
  driver counters at sub-100ms polling resolution, and maps 1:1 onto real
  DCGM field IDs -- so this collector is a drop-in conceptual replacement
  documented against production DCGM field names for portability.
"""

import csv
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import pynvml


DCGM_FIELD_MAP = {
    "gpu_util_pct": "DCGM_FI_DEV_GPU_UTIL",
    "mem_used_mb": "DCGM_FI_DEV_FB_USED",
    "power_watts": "DCGM_FI_DEV_POWER_USAGE",
    "sm_clock_mhz": "DCGM_FI_DEV_SM_CLOCK",
    "mem_copy_util_pct": "DCGM_FI_DEV_MEM_COPY_UTIL",
    "temperature_c": "DCGM_FI_DEV_GPU_TEMP",
}


@dataclass
class GPUSample:
    timestamp: float
    gpu_util_pct: float
    mem_used_mb: float
    mem_total_mb: float
    power_watts: float
    sm_clock_mhz: float
    mem_copy_util_pct: float
    temperature_c: float


class GPUMonitor:
    """
    Background-thread GPU telemetry poller.

    poll_interval_ms default of 200ms (see config.py MonitoringConfig)
    balances sampling resolution against CPU thread contention with the
    benchmark's own request-generation loop -- both compete for the
    single Colab CPU core allocation.
    """

    def __init__(self, poll_interval_ms: int = 200, device_index: int = 0):
        self.poll_interval_s = poll_interval_ms / 1000.0
        self.device_index = device_index
        self._samples: List[GPUSample] = []
        self._stop_event = threading.Event()
        self._thread = None

        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        name = pynvml.nvmlDeviceGetName(self._handle)
        self.gpu_name = name if isinstance(name, str) else name.decode()

    def _poll_loop(self):
        while not self._stop_event.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                clock = pynvml.nvmlDeviceGetClockInfo(
                    self._handle, pynvml.NVML_CLOCK_SM
                )
                temp = pynvml.nvmlDeviceGetTemperature(
                    self._handle, pynvml.NVML_TEMPERATURE_GPU
                )
                sample = GPUSample(
                    timestamp=time.time(),
                    gpu_util_pct=util.gpu,
                    mem_used_mb=mem.used / (1024 ** 2),
                    mem_total_mb=mem.total / (1024 ** 2),
                    power_watts=power_mw / 1000.0,
                    sm_clock_mhz=clock,
                    mem_copy_util_pct=util.memory,
                    temperature_c=temp,
                )
                self._samples.append(sample)
            except pynvml.NVMLError:
                pass
            time.sleep(self.poll_interval_s)

    def start(self):
        self._samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> List[GPUSample]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self._samples

    def summary(self) -> dict:
        if not self._samples:
            return {}
        util = [s.gpu_util_pct for s in self._samples]
        mem = [s.mem_used_mb for s in self._samples]
        power = [s.power_watts for s in self._samples]
        tensor_active_note = (
            "PIPE_TENSOR_ACTIVE unavailable via pynvml on consumer/T4 "
            "driver builds -- requires DCGM daemon on A100/H100 with "
            "MIG or full profiling permissions. See README GPU Metrics "
            "section for the production Prometheus+DCGM equivalent query."
        )
        return {
            "gpu_name": self.gpu_name,
            "n_samples": len(self._samples),
            "gpu_util_pct_mean": sum(util) / len(util),
            "gpu_util_pct_max": max(util),
            "mem_used_mb_mean": sum(mem) / len(mem),
            "mem_used_mb_max": max(mem),
            "power_watts_mean": sum(power) / len(power),
            "power_watts_max": max(power),
            "tensor_core_note": tensor_active_note,
        }

    def save_csv(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self._samples[0]).keys()))
            writer.writeheader()
            for s in self._samples:
                writer.writerow(asdict(s))
