"""Evaluation core: gridding, supervised windows, rolling-origin splits with purge, MASE, baselines.

Design rules
  * Windows never cross segment boundaries (no imputation across gaps).
  * Training data for a split ends at the origin; test targets start after a purge gap
    (>= horizon + lag window) so no training window overlaps test information.
  * MASE denominator = in-sample persistence MAE at the same horizon on the training data,
    so MASE < 1 means "better than persistence".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import pandas as pd


def to_grid(df: pd.DataFrame, cadence_min: int, channels: list[str], max_gap_min: float = 5.0) -> pd.DataFrame:
    """Resample each segment to a regular grid by mean; small holes (<= max_gap) are left NaN and dropped later.

    Returns columns: ts, segment, <channels>. Rows with NaN are removed and segments re-split on holes.
    """
    parts = []
    for _, seg in df.groupby("segment"):
        g = seg.set_index("ts")[channels].resample(f"{cadence_min}min").mean().dropna()
        if len(g):
            parts.append(g.reset_index())
    out = pd.concat(parts, ignore_index=True)
    gap = out["ts"].diff().dt.total_seconds().div(60)
    # a new segment starts when the step is larger than the cadence allows
    limit = max(max_gap_min, cadence_min * 1.5)
    out["segment"] = (gap > limit).cumsum().astype(int)
    return out


@dataclass
class Windows:
    X: np.ndarray        # (n, n_lags * n_channels)
    y: np.ndarray        # (n,) target value at t + h
    y0: np.ndarray       # (n,) target value at t (persistence forecast)
    t0: np.ndarray       # (n,) origin timestamp (datetime64)
    t_target: np.ndarray # (n,) target timestamp


def make_windows(grid: pd.DataFrame, target: str, channels: list[str], n_lags: int, horizon_steps: int) -> Windows:
    Xs, ys, y0s, t0s, tts = [], [], [], [], []
    ch = grid[channels].to_numpy(dtype=float)
    tgt = grid[target].to_numpy(dtype=float)
    ts = grid["ts"].to_numpy()
    for _, idx in grid.groupby("segment").indices.items():
        idx = np.sort(idx)
        n = len(idx)
        for i in range(n_lags - 1, n - horizon_steps):
            lo = idx[i - n_lags + 1]
            hi = idx[i]
            tgt_i = idx[i + horizon_steps]
            Xs.append(ch[lo : hi + 1].ravel())
            ys.append(tgt[tgt_i])
            y0s.append(tgt[hi])
            t0s.append(ts[hi])
            tts.append(ts[tgt_i])
    if not Xs:
        return Windows(np.empty((0, n_lags * len(channels))), np.empty(0), np.empty(0), np.empty(0, "datetime64[ns]"), np.empty(0, "datetime64[ns]"))
    return Windows(np.array(Xs), np.array(ys), np.array(y0s), np.array(t0s), np.array(tts))


def mase(y_true: np.ndarray, y_pred: np.ndarray, train_y: np.ndarray, train_y0: np.ndarray) -> float:
    """MASE relative to persistence: MAE(model, test) / MAE(persistence, train)."""
    denom = np.mean(np.abs(train_y - train_y0))
    if denom == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def rolling_origin(
    w: Windows,
    train_days: float,
    test_days: float,
    purge_minutes: float,
    step_days: float | None = None,
    start: np.datetime64 | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) with a growing window anchored at `start`.

    The first split trains on `train_days` of data; each next origin advances by `step_days`
    (default = test_days). Test windows begin `purge_minutes` after the train end (by target time of
    train windows, which is what overlaps).
    """
    if len(w.t0) == 0:
        return
    step_days = test_days if step_days is None else step_days
    t_start = w.t0.min() if start is None else start
    t_last = w.t_target.max()
    day = np.timedelta64(1, "D")
    purge = np.timedelta64(int(purge_minutes * 60), "s")
    origin = t_start + np.timedelta64(int(train_days * 86400), "s")
    while True:
        test_lo = origin + purge
        test_hi = test_lo + np.timedelta64(int(test_days * 86400), "s")
        if test_lo >= t_last:
            break
        # train windows: both origin and *target* must fall before the train end
        train_idx = np.where((w.t0 >= t_start) & (w.t_target <= origin))[0]
        test_idx = np.where((w.t0 >= test_lo) & (w.t_target < test_hi))[0]
        if len(train_idx) and len(test_idx):
            yield train_idx, test_idx
        origin = origin + np.timedelta64(int(step_days * 86400), "s")


# ---- baselines (no training) -------------------------------------------------

def persistence(w: Windows, idx: np.ndarray) -> np.ndarray:
    return w.y0[idx]


def moving_average(w: Windows, idx: np.ndarray, n_channels: int, target_pos: int = 0) -> np.ndarray:
    """Mean of the target channel over the lag window. target_pos = position of target among channels."""
    X = w.X[idx]
    n_lags = X.shape[1] // n_channels
    cols = [lag * n_channels + target_pos for lag in range(n_lags)]
    return X[:, cols].mean(axis=1)


def evaluate_split(
    w: Windows, train_idx: np.ndarray, test_idx: np.ndarray, predict: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> dict:
    """predict(train_idx, test_idx) -> predictions for test_idx. Returns metrics dict."""
    y_pred = predict(train_idx, test_idx)
    y_true = w.y[test_idx]
    return {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "mase": mase(y_true, y_pred, w.y[train_idx], w.y0[train_idx]),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }
