"""Cell-by-cell comparison of the weekly-pattern run (batch 4, --weekly) with the same models without it.

DALTON eco2: batch 4 vs batch 3 (ridge, huber, policy; same devices, splits and seeds). Urbino eco2: batch 4 vs batch 2.
Per model, D and horizon: share of devices meeting the cold-start criterion and median over devices of the geometric
mean rel_mae, with and without --weekly. weekly_naive (last week's change over the same interval) exists only in
batch 4. Urbino also by calendar day type (check_urbino.py --calendar-only outputs, device 'all': geometric mean with CI).
Reads summaries only.

  python scripts/compare_weekly.py --out results/checks/weekly_comparison
  -> results/checks/weekly_comparison_{dalton,urbino,urbino_daytype}.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

MODELS = ["ridge", "huber", "policy", "weekly_naive"]
CELL = ["horizon_min", "model", "train_days"]


def by_cell(prefix: str) -> pd.DataFrame:
    frac = pd.read_csv(f"{prefix}_fraction.csv")
    cells = pd.read_csv(f"{prefix}_cells.csv")
    med = cells.groupby(CELL)["geo_rel_mae"].median().rename("median_geo_rel_mae")
    out = frac.set_index(CELL)[["n_devices", "frac_meeting"]].join(med).reset_index()
    return out[out["model"].isin(MODELS)]


def compare(weekly: str, base: str) -> pd.DataFrame:
    w, b = by_cell(weekly), by_cell(base)
    m = w.merge(b, on=CELL, how="left", suffixes=("_weekly", "_base"))
    m["d_frac_meeting"] = m["frac_meeting_weekly"] - m["frac_meeting_base"]
    m["d_median_geo"] = m["median_geo_rel_mae_weekly"] - m["median_geo_rel_mae_base"]
    return m.sort_values(CELL)


def compare_daytype(weekly: str, base: str) -> pd.DataFrame:
    keys = CELL + ["day_type"]
    cols = ["n_days", "geo_rel_mae", "geo_ci_lo", "geo_ci_hi", "pooled_rel_mae"]
    w, b = (pd.read_csv(f) for f in (weekly, base))
    w, b = (d[(d["device"] == "all") & d["model"].isin(MODELS)][keys + cols] for d in (w, b))
    m = w.merge(b, on=keys, how="left", suffixes=("_weekly", "_base"))
    m["d_geo"] = m["geo_rel_mae_weekly"] - m["geo_rel_mae_base"]
    return m.sort_values(keys)


def _pair(weekly: pd.Series, base: pd.Series, fmt: str) -> pd.Series:
    """'weekly / base' for printing; '-' where a side does not exist (weekly_naive has no base)."""
    f = lambda s: s.map(lambda v: "-" if pd.isna(v) else fmt.format(v))  # noqa: E731
    return f(weekly) + " / " + f(base)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/checks/weekly_comparison")
    a = p.parse_args()
    for name, weekly, base in (("dalton", "results/summary_batch4_dalton", "results/summary_batch3_dalton"),
                               ("urbino", "results/summary_batch4_urbino", "results/summary_batch2_urbino")):
        c = compare(weekly, base)
        c.to_csv(f"{a.out}_{name}.csv", index=False)
        print(f"\n== {name}: share meeting (weekly / base) and median geo rel_mae (weekly / base)")
        c = c.assign(frac=_pair(100 * c["frac_meeting_weekly"], 100 * c["frac_meeting_base"], "{:.0f}"),
                     geo=_pair(c["median_geo_rel_mae_weekly"], c["median_geo_rel_mae_base"], "{:.3f}"))
        print(c.pivot(index=["horizon_min", "model"], columns="train_days", values="frac")
              .to_string())
        print(c.pivot(index=["horizon_min", "model"], columns="train_days", values="geo")
              .to_string())
    dt = compare_daytype("results/checks/urbino_weekly_daytype_calendar.csv",
                         "results/checks/urbino_daytype_calendar.csv")
    dt.to_csv(f"{a.out}_urbino_daytype.csv", index=False)
    dt = dt.assign(geo=_pair(dt["geo_rel_mae_weekly"], dt["geo_rel_mae_base"], "{:.3f}"))
    print("\n== urbino by day type: geometric mean rel_mae (weekly / base), device all")
    print(dt.pivot(index=["horizon_min", "day_type", "model"], columns="train_days", values="geo").to_string())


if __name__ == "__main__":
    main()
