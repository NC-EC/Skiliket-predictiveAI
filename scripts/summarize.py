"""Summarize runner CSVs: one row per (dataset, device, target, horizon, model, D), plus D* per model.

D* = the smallest training length D that meets the cold-start criterion (stats.meets_cold_start: GEOMETRIC mean of
rel_mae < 1 and upper 95% block-bootstrap CI bound < 1, splits as blocks so the seeds of one split stay together).
NaN if no D meets it.

Short test splits are dropped first: a split whose test set has fewer than --min-test-frac (default 0.5) of the
windows of a full test day for that device and horizon (the largest n_test seen) is excluded, because a handful of
windows with a near-zero persistence error gives an unstable ratio (seen on DALTON H1_41: 10 windows, rel_mae 33).
The rule is fixed before looking at results; n_splits_dropped reports its effect.

Also reported: arithmetic mean rel_mae with CI (secondary) and pooled rel_mae = sum of model absolute errors /
sum of persistence absolute errors over all kept splits (robustness check; needs the mae and n_test columns).

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

from coldstart.stats import bootstrap_ci, geomean_ci

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


def drop_short_splits(df: pd.DataFrame, min_test_frac: float = 0.5) -> tuple[pd.DataFrame, pd.Series]:
    """Remove splits whose test set is shorter than min_test_frac of a full test day (largest n_test per device and
    horizon). Returns the kept rows and a boolean Series (indexed like df) marking dropped rows."""
    if "n_test" not in df or min_test_frac <= 0:
        return df, pd.Series(False, index=df.index)
    full = df.groupby(["dataset", "device", "target", "horizon_min"])["n_test"].transform("max")
    short = df["n_test"] < min_test_frac * full
    return df[~short], short


def _pooled(g: pd.DataFrame, pers: pd.DataFrame | None) -> float:
    if pers is None or "mae" not in g or "n_test" not in g:
        return np.nan
    m = g.groupby("split").agg(mae=("mae", "mean"), n=("n_test", "first"))
    p = pers.groupby("split")["mae"].mean().reindex(m.index)
    denom = (p * m["n"]).sum()
    return float((m["mae"] * m["n"]).sum() / denom) if denom > 0 else np.nan


def summarize_cells(df: pd.DataFrame, n_resamples: int = 1000, min_test_frac: float = 0.5) -> pd.DataFrame:
    df, short = drop_short_splits(df, min_test_frac)
    rows = []
    cell_keys = KEYS[:-1] + ["train_days"]
    pers_by_cell = {k: g for k, g in df[df["model"] == "persistence"].groupby(cell_keys, sort=False)}
    for key, g in df.groupby(KEYS + ["train_days"], sort=True):
        # a split wins when its seed-mean rel_mae is below persistence's (= 1 by definition)
        per_split = g.groupby("split")["rel_mae"].mean()
        geo, glo, ghi = geomean_ci(g["rel_mae"], n_resamples=n_resamples, groups=g["split"])
        mean, lo, hi = bootstrap_ci(g["rel_mae"], n_resamples=n_resamples, groups=g["split"])
        ck = tuple(v for k, v in zip(KEYS + ["train_days"], key) if k != "model")
        rows.append({**dict(zip(KEYS + ["train_days"], key)),
                     "n_splits": len(per_split), "n_rows": len(g),
                     "median_rel_mae": float(g["rel_mae"].median()),
                     "geo_rel_mae": geo, "geo_ci_lo": glo, "geo_ci_hi": ghi,
                     "mean_rel_mae": mean, "ci_lo": lo, "ci_hi": hi,
                     "pooled_rel_mae": _pooled(g, pers_by_cell.get(ck)),
                     "pct_splits_won": float(100 * (per_split < 1).mean()),
                     "meets_cold_start": bool(geo < 1 and ghi < 1),
                     **{new: float(g[c].median()) if c in g else np.nan for c, new in COST.items()},
                     "parallel": float(g["parallel"].max()) if "parallel" in g else np.nan})
    out = pd.DataFrame(rows)
    out.attrs["rows_dropped_short_test"] = int(short.sum())
    return out


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
    p.add_argument("--min-test-frac", type=float, default=0.5,
                   help="drop test splits shorter than this fraction of a full test day (0 disables)")
    a = p.parse_args()
    paths = sorted({f for pat in a.csv for f in (glob.glob(pat) or [pat])})
    cells = summarize_cells(load(paths), a.resamples, a.min_test_frac)
    print(f"rows dropped (short test splits): {cells.attrs.get('rows_dropped_short_test', 0)}")
    ds = d_star(cells)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(f"{a.out}_cells.csv", index=False)
    ds.to_csv(f"{a.out}_dstar.csv", index=False)
    print(ds.to_string(index=False))


if __name__ == "__main__":
    main()
