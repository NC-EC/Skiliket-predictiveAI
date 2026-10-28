"""Compute-cost measurement: wall time, CPU time, peak memory, model size, inference latency."""
from __future__ import annotations

import pickle
import sys
import time
import tracemalloc
from dataclasses import dataclass, asdict


def _max_rss_mb() -> float:
    """Process peak resident memory in MB. `resource` is Unix-only, so Windows uses the peak working set."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                                                     wintypes.DWORD]
        kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return counters.PeakWorkingSetSize / 2**20
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 1024  # bytes on macOS, KB on Linux


@dataclass
class Cost:
    wall_s: float
    cpu_s: float
    peak_py_mb: float      # tracemalloc peak (Python allocations; numpy buffers are included)
    max_rss_mb: float      # process high-water mark (monotonic over the process lifetime)
    model_bytes: int | None = None
    infer_ms: float | None = None

    def asdict(self) -> dict:
        return asdict(self)


def measure_fit(fit, *args, **kwargs):
    """Run fit(*args) and return (result, Cost). Use a fresh process per configuration for clean max_rss."""
    tracemalloc.start()
    w0, c0 = time.perf_counter(), time.process_time()
    result = fit(*args, **kwargs)
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cost = Cost(wall, cpu, peak / 1e6, _max_rss_mb())
    try:
        cost.model_bytes = len(pickle.dumps(result))
    except Exception:
        pass
    return result, cost


def measure_inference(model, X, repeats: int = 50) -> float:
    """Median latency in ms for a single-row prediction."""
    import statistics

    row = X[:1]
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        model.predict(row)
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)
