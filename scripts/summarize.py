"""Summarize runner CSVs: one row per (dataset, device, target, horizon, model, D), plus D* per model.

D* = the smallest training length D that meets the cold-start criterion (stats.meets_cold_start: mean rel_mae < 1 and
upper 95% block-bootstrap CI bound < 1, splits as blocks so the seeds of one split stay together). NaN if no D meets it.

Cost columns are medians over the cell: train_s (wall time of one fit), peak_py_mb (tracemalloc peak during the fit:
the memory figure for the paper, not the process RSS, which is dominated by loading the data), infer_ms per window,
model_bytes. They are timing only when the run had parallel == 1.

Example:
  python scripts/summarize.py results/runs/dalton_*.csv --out results/summary_dalton
  -> results/summary_dalton_cells.csv, results/summary_dalton_dstar.csv
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from coldstart.stats import bootstrap_ci

KEYS = ["dataset", "device", "target", "horizon_min", "model"]
COST = {"wall_s": "train_s", "peak_py_mb": "peak_py_mb", "infer_ms": "infer_ms", "model_bytes": "model_bytes"}


def load(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if "device" not in df:  # CSVs written before the runner had a device column: use the file name
            df["device"] = Path(p).stem
        df["device"] = df["device"].fillna("").astype(str)
        df["source"] = Path(p).name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize_cells(df: pd.DataFrame, n_resamples: int = 1000) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(KEYS + ["train_days"], sort=True):
        # a split wins when its seed-mean rel_mae is below persistence's (= 1 by definition)
        per_split = g.groupby("split")["rel_mae"].mean()
        mean, lo, hi = bootstrap_ci(g["rel_mae"], n_resamples=n_resamples, groups=g["split"])
        rows.append({**dict(zip(KEYS + ["train_days"], key)),
                     "n_splits": len(per_split), "n_rows": len(g),
                     "median_rel_mae": float(g["rel_mae"].median()),
                     "mean_rel_mae": mean, "ci_lo": lo, "ci_hi": hi,
                     "pct_splits_won": float(100 * (per_split < 1).mean()),
                     "meets_cold_start": bool(mean < 1 and hi < 1),
                     **{new: float(g[c].median()) if c in g else np.nan for c, new in COST.items()},
                     "parallel": float(g["parallel"].max()) if "parallel" in g else np.nan})
    return pd.DataFrame(rows)


def d_star(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in cells.groupby(KEYS, sort=True):
        ok = g.loc[g["meets_cold_start"], "train_days"]
        rows.append({**dict(zip(KEYS, key)), "d_star": float(ok.min()) if len(ok) else np.nan,
                     "days_tested": " ".join(f"{d:g}" for d in sorted(g["train_days"]))})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="+", help="runner CSVs (globs allowed)")
    p.add_argument("--out", required=True, help="output prefix, e.g. results/summary_dalton")
    p.add_argument("--resamples", type=int, default=1000)
    a = p.parse_args()
    paths = sorted({f for pat in a.csv for f in (glob.glob(pat) or [pat])})
    cells = summarize_cells(load(paths), a.resamples)
    ds = d_star(cells)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(f"{a.out}_cells.csv", index=False)
    ds.to_csv(f"{a.out}_dstar.csv", index=False)
    print(ds.to_string(index=False))


if __name__ == "__main__":
    main()
