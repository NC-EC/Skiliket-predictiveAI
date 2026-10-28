"""Compute-cost measurement: wall time, CPU time, peak memory, model size, inference latency."""
from __future__ import annotations

import pickle
import resource
import time
import tracemalloc
from dataclasses import dataclass, asdict


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
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux
    cost = Cost(wall, cpu, peak / 1e6, rss_kb / 1024)
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
