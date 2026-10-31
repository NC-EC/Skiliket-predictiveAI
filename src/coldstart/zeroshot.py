"""Pretrained zero-shot forecasters (Chronos family) as the D = 0 point of the learning curve.

A zero-shot model is never trained on the kit's data: at each forecast origin it receives only the last `context`
grid steps of the target (and, for Chronos-2, optionally the covariates), all inside the origin's segment, and
forecasts the next steps. So it needs only `context` steps of history, e.g. 2 h or 24 h, instead of D days. No
reading is filled in: a context that would cross a segment boundary is cut at the boundary (shorter context).

The pipeline object only needs `predict_quantiles(inputs, prediction_length, quantile_levels)`, as in
chronos-forecasting >= 2.0 (`BaseChronosPipeline`). Bolt returns a tensor (batch, steps, quantiles); Chronos-2 returns
a list of tensors (variates, steps, quantiles); both are handled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def contexts(grid: pd.DataFrame, target: str, t0: np.ndarray, n_steps: int, covariates: list[str] | None = None):
    """For each origin time in t0, the last n_steps values of `target` up to and including t0, within t0's segment.

    Returns a list of 1-D float arrays (target only) or, with covariates, a list of dicts
    {"target": arr, "past_covariates": {name: arr}} as Chronos-2 expects."""
    ts = grid["ts"].to_numpy()
    pos = pd.Series(np.arange(len(grid)), index=pd.DatetimeIndex(ts))
    seg = grid["segment"].to_numpy()
    y = grid[target].to_numpy(dtype=float)
    cov = {c: grid[c].to_numpy(dtype=float) for c in (covariates or [])}
    # first row of each segment, to cut contexts at the boundary
    seg_start = np.zeros(len(grid), dtype=int)
    starts = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1]
    for a, b in zip(starts, np.r_[starts[1:], len(grid)]):
        seg_start[a:b] = a
    out = []
    for t in pd.DatetimeIndex(t0):
        i = int(pos[t])
        lo = max(seg_start[i], i - n_steps + 1)
        if covariates:
            out.append({"target": y[lo:i + 1], "past_covariates": {c: v[lo:i + 1] for c, v in cov.items()}})
        else:
            out.append(y[lo:i + 1])
    return out


def median_forecast(pipe, ctx: list, prediction_length: int, batch_size: int = 256) -> np.ndarray:
    """(n, prediction_length) median forecasts of the target for each context."""
    import torch

    rows = []
    for k in range(0, len(ctx), batch_size):
        chunk = ctx[k:k + batch_size]
        inputs = chunk if isinstance(chunk[0], dict) else [torch.tensor(c, dtype=torch.float32) for c in chunk]
        q, _ = pipe.predict_quantiles(inputs, prediction_length=prediction_length, quantile_levels=[0.5])
        if isinstance(q, (list, tuple)):  # Chronos-2: one (variates, steps, quantiles) tensor per series
            rows += [np.asarray(x[0, :, 0], dtype=float) for x in q]
        else:  # Bolt / Chronos: (batch, steps, quantiles)
            rows += list(np.asarray(q[:, :, 0], dtype=float))
    return np.vstack(rows) if rows else np.empty((0, prediction_length))


def n_params(pipe) -> int | None:
    m = getattr(pipe, "model", None) or getattr(pipe, "inner_model", None)
    try:
        return int(sum(p.numel() for p in m.parameters()))
    except Exception:  # noqa: BLE001 - size is informational
        return None
