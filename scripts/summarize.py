"""Summarize runner CSVs: one row per (dataset, device, target, horizon, model, D), plus D* per model.

Cold-start criterion (stats.meets_cold_start): GEOMETRIC mean of rel_mae < 1 and upper 95% block-bootstrap CI bound
< 1, splits as blocks so the seeds of one split stay together.

D* (stable) = the smallest training length D that meets the criterion and keeps meeting it at every larger D the device
was tested on. d_star_status:
  "reached"      stable D* confirmed by at least one larger D;
  "unconfirmed"  D* is the largest D tested (meets there, nothing larger to confirm it); d_star is kept;
  "insufficient" the device was tested at a single D, so stability cannot be judged; d_star is NaN;
  "never"        tested at the largest D of the design and no stable D exists; d_star is NaN;
  "untestable"   too few days for the largest D of the design, and no stable D within the D it has; d_star is NaN.
d_star_first
(smallest D that meets it once) is kept for comparison. {out}_fraction.csv: share of devices meeting the criterion at
each D among the devices tested at that D (the main figure).

Policy: if the CSVs carry the models' own validation scores (cv_rel_mae column), the policy rows are recomputed here
from POLICY_CANDIDATES (persistence when no candidate validates below 1), so the candidate set can change without a
re-run. Older CSVs without that column keep the runner's policy rows.

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
POLICY_CANDIDATES = ("ridge", "huber")  # LightGBM's score is optimistic: validated on its own early-stopping tail
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


def recompute_policy(df: pd.DataFrame, candidates=POLICY_CANDIDATES) -> pd.DataFrame:
    """Replace the policy rows: per split, the candidate (seed 0) with the lowest cv_rel_mae if it is < 1, else
    persistence. Metrics come from the chosen model's row; time is the sum over the candidates (memory the max)."""
    if "cv_rel_mae" not in df:
        return df
    split_keys = ["dataset", "device", "target", "horizon_min", "train_days", "split"]
    base = df[df["model"] != "policy"]
    cand = base[base["model"].isin(candidates) & (base["seed"] == 0)]
    pers = {k: g.iloc[0] for k, g in base[base["model"] == "persistence"].groupby(split_keys, sort=False)}
    rows = []
    for key, g in cand.groupby(split_keys, sort=False):
        scored = g[np.isfinite(g["cv_rel_mae"])]
        best = scored.loc[scored["cv_rel_mae"].idxmin()] if len(scored) else None
        win = best is not None and best["cv_rel_mae"] < 1
        row = best if win else pers.get(key)
        if row is None:
            continue
        row = row.copy()
        row["model"], row["seed"] = "policy", 0
        row["policy_choice"] = best["model"] if win else "persistence"
        for c in ("wall_s", "cpu_s"):
            if c in g:
                row[c] = g[c].sum()
        if "peak_py_mb" in g:
            row["peak_py_mb"] = g["peak_py_mb"].max()
        rows.append(row)
    return pd.concat([base, pd.DataFrame(rows)], ignore_index=True)


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
    df, short = drop_short_splits(recompute_policy(df), min_test_frac)
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
    design_max = cells.groupby(["dataset", "target", "horizon_min"])["train_days"].max()
    rows = []
    for key, g in cells.groupby(KEYS, sort=True):
        g = g.sort_values("train_days")
        meets = g["meets_cold_start"].to_numpy(bool)
        days = g["train_days"].to_numpy(float)
        # stable: meets here and at every larger D tested (suffix of all-True)
        stable = np.flip(np.logical_and.accumulate(np.flip(meets)))
        k = dict(zip(KEYS, key))
        ds = float(days[stable][0]) if stable.any() else np.nan
        if len(days) < 2:
            ds, status = np.nan, "insufficient"
        elif np.isfinite(ds):
            status = "unconfirmed" if ds == days.max() else "reached"
        else:
            status = "never" if days.max() >= design_max[k["dataset"], k["target"], k["horizon_min"]] else "untestable"
        rows.append({**k, "d_star": ds, "d_star_status": status,
                     "d_star_first": float(days[meets][0]) if meets.any() else np.nan,
                     "max_days_tested": float(days.max()),
                     "days_tested": " ".join(f"{d:g}" for d in days)})
    return pd.DataFrame(rows)


def fraction_meeting(cells: pd.DataFrame) -> pd.DataFrame:
    """Per dataset x target x horizon x model x D: devices tested at D, devices meeting the criterion, their share."""
    g = cells.groupby(["dataset", "target", "horizon_min", "model", "train_days"])["meets_cold_start"]
    out = g.agg(n_devices="size", n_meeting="sum").reset_index()
    out["frac_meeting"] = out["n_meeting"] / out["n_devices"]
    return out


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
    fraction_meeting(cells).to_csv(f"{a.out}_fraction.csv", index=False)
    print(ds.to_string(index=False))


if __name__ == "__main__":
    main()
