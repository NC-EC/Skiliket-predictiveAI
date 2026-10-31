"""Weekly-pattern inputs: day of week and the target's change over the same horizon one week earlier.

Kept out of evaluate.py on purpose: the grid cache key hashes evaluate.py, and these inputs do not change the grid.

For a window with origin t and target time t + h, `last_week_change` = y(t + h - 7 d) - y(t - 7 d): what happened over
the same clock interval one week before. University timetables repeat weekly within a term, so this tells a model
whether a class started at this hour last week. It is past data for a deployed kit only if the kit was already running
at t - 7 d, so `weekly_columns` takes the kit's start (the start of the D training days) and marks the value
unavailable before it: with D < 7 days no window has it, as on a real new kit. Unavailable values are encoded as 0 with
an availability flag (an input encoding; no sensor reading is filled in).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEEK = np.timedelta64(7, "D")
N_DOW = 7
WEEKLY = ["last_week_change", "last_week_available"] + [f"dow_{i}" for i in range(N_DOW)]


def last_week_change(grid: pd.DataFrame, target: str, t0: np.ndarray, t_target: np.ndarray) -> np.ndarray:
    """y(t_target - 7 d) - y(t0 - 7 d) from the grid; NaN where either grid point is missing."""
    s = grid.set_index("ts")[target]
    s = s[~s.index.duplicated()]
    a = s.reindex(pd.DatetimeIndex(t_target - WEEK)).to_numpy(dtype=float)
    b = s.reindex(pd.DatetimeIndex(t0 - WEEK)).to_numpy(dtype=float)
    return a - b


def day_of_week(t0: np.ndarray) -> np.ndarray:
    """(n, 7) one-hot day of week of the window origin (Monday = column 0), in the source's clock."""
    d = pd.DatetimeIndex(t0).day_of_week.to_numpy()
    return np.eye(N_DOW)[d]


def weekly_columns(change: np.ndarray, t0: np.ndarray, kit_start: np.datetime64) -> np.ndarray:
    """(n, 9): last-week change (0 where unavailable), availability flag, day-of-week one-hot. A value is available
    only if it exists in the grid and t0 - 7 d is not before the kit's start."""
    avail = np.isfinite(change) & ((t0 - WEEK) >= kit_start)
    return np.column_stack([np.where(avail, change, 0.0), avail.astype(float), day_of_week(t0)])


def weekly_naive(y0: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Baseline: persistence plus last week's change over the same interval where available, else persistence."""
    return y0 + cols[:, 0] * cols[:, 1]
