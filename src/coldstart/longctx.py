"""Long-context inputs for the local models: a summary of the last 24 hours of the target.

Why: the pretrained models (zeroshot.py) see 24 h of the target at each origin, while the local models see a 30-min
lag window plus calendar features. These inputs give the local models the same 24 h, so the comparison separates the
effect of the longer context from the effect of pretraining.

For an origin t with last reading y0 = y(t), the inputs are
- 24 hourly means of the target over (t - (k + 1) h, t - k h], k = 0..23, each minus y0 (level relative to now);
- the share of those hourly bins that hold at least one reading;
- yesterday's change over the same clock interval, y(t + h - 24 h) - y(t - 24 h), and its availability flag.
Only readings at or after the kit's start (the first of its D training days) count: a deployed kit has no readings
from before it was switched on. Readings are taken by time stamp, across segment gaps; a missing reading is skipped,
never filled. An empty bin or an unavailable change is encoded as 0 with the share / flag telling the model so (an
input encoding; no sensor reading is filled in). Kept out of evaluate.py so the grid cache key does not change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOURS = 24
LONGCTX = [f"ctx_mean_h{k}" for k in range(HOURS)] + ["ctx_share_available", "yesterday_change", "yesterday_available"]


class LongContext:
    """Precomputes the target on a regular time index (NaN where the grid has no reading) for one device."""

    def __init__(self, grid: pd.DataFrame, target: str, cadence_min: int, hours: int = HOURS):
        s = grid.set_index("ts")[target]
        s = s[~s.index.duplicated()].sort_index()
        self.idx = pd.date_range(s.index.min(), s.index.max(), freq=pd.Timedelta(minutes=cadence_min))
        self.v = s.reindex(self.idx).to_numpy(dtype=float)
        self.per_hour = int(round(60 / cadence_min))
        self.hours = hours

    def columns(self, t0: np.ndarray, y0: np.ndarray, t_target: np.ndarray, kit_start) -> np.ndarray:
        """(n, hours + 3) inputs for windows with origins t0 (last reading y0) and target times t_target."""
        v = np.where(self.idx.to_numpy() >= np.datetime64(kit_start), self.v, np.nan)
        ok = np.isfinite(v)
        csum = np.concatenate([[0.0], np.cumsum(np.where(ok, v, 0.0))])
        ccnt = np.concatenate([[0], np.cumsum(ok)])
        pos = self.idx.get_indexer(pd.DatetimeIndex(t0))
        if (pos < 0).any():
            raise ValueError("an origin time is not on the cadence grid")
        n, b = len(pos), self.per_hour
        means, avail = np.zeros((n, self.hours)), np.zeros((n, self.hours))
        for k in range(self.hours):
            hi = np.clip(pos + 1 - k * b, 0, len(v))  # bin k = grid points (pos - (k + 1) b, pos - k b]
            lo = np.clip(hi - b, 0, len(v))
            cnt = ccnt[hi] - ccnt[lo]
            has = cnt > 0
            means[has, k] = (csum[hi] - csum[lo])[has] / cnt[has] - y0[has]
            avail[:, k] = has
        day = 24 * b
        a_pos, b_pos = pos - day, self.idx.get_indexer(pd.DatetimeIndex(t_target)) - day
        inside = (a_pos >= 0) & (b_pos >= 0)
        change = np.full(n, np.nan)
        change[inside] = v[b_pos[inside]] - v[a_pos[inside]]
        y_ok = np.isfinite(change)
        return np.column_stack([means, avail.mean(axis=1), np.where(y_ok, change, 0.0), y_ok.astype(float)])
