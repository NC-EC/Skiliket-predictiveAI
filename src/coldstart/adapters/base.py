from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema import CANONICAL_COLUMNS, CHANNELS


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
