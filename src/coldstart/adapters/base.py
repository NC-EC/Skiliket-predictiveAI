from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema import CANONICAL_COLUMNS, CHANNELS

# Physical plausibility limits for the canonical channels; readings outside are invalid (dropped, never clipped).
PLAUSIBLE = {
    "eco2": (0.0, 40000.0),    # exclusive lower bound: 0 ppm is a failed read
    "tvoc": (-1e-9, 1e6),      # 0 ppb is a legitimate MOx floor reading
    "temp_c": (-40.0, 60.0),
    "rh_pct": (0.0, 100.0),    # exclusive lower bound: 0 % is a failed read
}


def finalize(df: pd.DataFrame, valid: pd.Series, max_gap_minutes: float) -> pd.DataFrame:
    """Sort, drop invalid rows from the usable stream, assign segment ids on gaps.

    Invalid rows are removed from the series (not imputed); the gap they leave splits
    the series when it exceeds max_gap_minutes. The `valid` column is kept (all True after
    dropping) so downstream code can assert it; the count of dropped rows goes to metadata.
    """
    df = df.copy()
    df["valid"] = valid.values
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    df = df[df["valid"]].reset_index(drop=True)
    gap = df["ts"].diff().dt.total_seconds().div(60)
    df["segment"] = (gap > max_gap_minutes).cumsum().astype(int)
    for c in CHANNELS:
        if c not in df.columns:
            df[c] = np.nan
    return df[CANONICAL_COLUMNS]


def median_step_s(ts: pd.Series) -> float:
    """Median sampling step of the raw time stamps, in seconds."""
    return float(ts.sort_values().diff().dt.total_seconds().median())


def default_max_gap(ts: pd.Series) -> float:
    """Gap (min) that splits a series: 5 min, widened to 1.5x the native step for coarser devices.

    Mirrors evaluate.to_grid, so a 10-min or hourly device is not shattered into one-row segments.
    """
    return max(5.0, 1.5 * median_step_s(ts) / 60)


def invalid_reasons(df: pd.DataFrame, channels: list[str]) -> dict[str, pd.Series]:
    """Boolean masks (True = invalid) for missing values, implausible values and duplicated time stamps."""
    reasons = {}
    for c in channels:
        lo, hi = PLAUSIBLE[c]
        reasons[f"{c}_missing"] = df[c].isna()
        reasons[f"{c}_out_of_range"] = df[c].notna() & ~((df[c] > lo) & (df[c] <= hi))
    # we cannot tell which copy of a duplicated time stamp is right, so all copies are dropped
    reasons["duplicate_ts"] = df["ts"].duplicated(keep=False)
    return reasons


def build_report(rows_raw: int, reasons: dict[str, pd.Series], out: pd.DataFrame, max_gap_minutes: float,
                 step_s: float) -> dict:
    invalid = pd.concat(reasons, axis=1).any(axis=1) if reasons else pd.Series(dtype=bool)
    return {
        "rows_raw": int(rows_raw),
        "rows_invalid_dropped": int(invalid.sum()),
        "invalid_by_reason": {k: int(v.sum()) for k, v in reasons.items() if v.any()},
        "rows_usable": int(len(out)),
        "segments": int(out["segment"].nunique()),
        "start": out["ts"].min(),
        "end": out["ts"].max(),
        "span_days": float((out["ts"].max() - out["ts"].min()).total_seconds() / 86400),
        "native_step_s": step_s,
        "max_gap_minutes": max_gap_minutes,
    }
