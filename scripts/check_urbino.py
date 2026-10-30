"""Urbino checks on finished runner CSVs (reads results/runs/urbino_*.csv only; no training).

1. Test days split into "class" and "no_class": a day is no_class when it falls on a weekend (midpoint of the test
   day, UTC) or when persistence's MAE on that day is below the 25th percentile of that room's test days (an empty
   room: CO2 barely moves). Per model, D and group: median and geometric mean of rel_mae over test days (seed mean
   per day) and the pooled rel_mae (sum of model abs errors / sum of persistence abs errors).
2. The rows summarize.py drops as short test splits: how many fall on class / no_class days, and n_test.
3. Policy recomputed from cv_rel_mae with threshold 0.95 instead of 1 (sensitivity only; the policy keeps 1).

The 25th-percentile threshold is taken over the kept (full) test days of each room and horizon and also applied to
the dropped days.

Control for check 1: the persistence-MAE rule uses rel_mae's own denominator, so it favours the models in "class".
--calendar-only repeats check 1 with the calendar alone: "weekday" vs "weekend_holiday" (weekend or Italian national
holiday of 2024), same columns.

  python scripts/check_urbino.py --out results/checks/urbino
  -> results/checks/urbino_daytype.csv, urbino_dropped.csv, urbino_policy_threshold.csv
  python scripts/check_urbino.py --out results/checks/urbino --calendar-only
  -> results/checks/urbino_daytype_calendar.csv
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from coldstart.stats import geomean_ci

_spec = importlib.util.spec_from_file_location("summarize", Path(__file__).parent / "summarize.py")
summarize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarize)

MODELS = ["huber", "ridge", "lightgbm", "random_forest", "gru", "policy"]
DAY = ["device", "horizon_min", "test_start"]
# Italian national holidays 2024 (Easter Monday 1 Apr); the Urbino rooms were recorded in 2024
IT_HOLIDAYS = {"01-01", "01-06", "04-01", "04-25", "05-01", "06-02", "08-15", "11-01", "12-08", "12-25", "12-26"}


def day_types(df: pd.DataFrame, kept: pd.DataFrame, q: float = 25, calendar_only: bool = False) -> pd.DataFrame:
    """One row per test day (device, horizon, test_start): persistence MAE, n_test, weekend flag and day type.
    calendar_only: "weekday" / "weekend_holiday" from the date alone (no persistence MAE, which is rel_mae's
    denominator and so would select days in the models' favour)."""
    pers = df[df["model"] == "persistence"].drop_duplicates(DAY)[DAY + ["mae", "n_test"]]
    pers = pers.rename(columns={"mae": "pers_mae"})
    mid = pd.to_datetime(pers["test_start"]) + pd.Timedelta(hours=12)
    pers["weekend"] = mid.dt.day_of_week >= 5
    if calendar_only:
        pers["holiday"] = mid.dt.strftime("%m-%d").isin(IT_HOLIDAYS) & (mid.dt.year == 2024)
        pers["day_type"] = np.where(pers["weekend"] | pers["holiday"], "weekend_holiday", "weekday")
        return pers
    kept_days = kept[kept["model"] == "persistence"].drop_duplicates(DAY)
    thr = kept_days.groupby(["device", "horizon_min"])["mae"].quantile(q / 100).rename("pers_mae_p25")
    pers = pers.join(thr, on=["device", "horizon_min"])
    pers["day_type"] = np.where(pers["weekend"] | (pers["pers_mae"] < pers["pers_mae_p25"]), "no_class", "class")
    return pers


def _scores(g: pd.DataFrame) -> dict:
    per_day = g.groupby("test_start").agg(rel=("rel_mae", "mean"), mae=("mae", "mean"), n=("n_test", "first"),
                                          pers=("pers_mae", "first"))
    geo, lo, hi = geomean_ci(g["rel_mae"], groups=g["test_start"], n_resamples=1000)
    return {"n_days": len(per_day), "median_rel_mae": float(per_day["rel"].median()), "geo_rel_mae": geo,
            "geo_ci_lo": lo, "geo_ci_hi": hi,
            "pooled_rel_mae": float((per_day["mae"] * per_day["n"]).sum() / (per_day["pers"] * per_day["n"]).sum())}


def by_day_type(kept: pd.DataFrame, days: pd.DataFrame) -> pd.DataFrame:
    d = kept[kept["model"].isin(MODELS)].merge(days[DAY + ["pers_mae", "day_type"]], on=DAY)
    rows = []
    for rooms, sub in [("all", d)] + [(r, g) for r, g in d.groupby("device")]:
        for key, g in sub.groupby(["horizon_min", "model", "train_days", "day_type"]):
            rows.append({"device": rooms, **dict(zip(["horizon_min", "model", "train_days", "day_type"], key)),
                         **_scores(g)})
    return pd.DataFrame(rows)


def policy_threshold(df: pd.DataFrame, min_test_frac: float, thresholds=(1.0, 0.95)) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        pol = summarize.recompute_policy(df, threshold=t)
        pol, _ = summarize.drop_short_splits(pol, min_test_frac)
        pol = pol[pol["model"] == "policy"]
        for key, g in pol.groupby(["device", "horizon_min", "train_days"]):
            geo, lo, hi = geomean_ci(g["rel_mae"], groups=g["split"], n_resamples=1000)
            rows.append({"threshold": t, **dict(zip(["device", "horizon_min", "train_days"], key)),
                         "n_days": len(g), "geo_rel_mae": geo, "geo_ci_hi": hi, "meets_cold_start": geo < 1 and hi < 1,
                         "median_rel_mae": float(g["rel_mae"].median()),
                         "pct_persistence": float(100 * (g["policy_choice"] == "persistence").mean()),
                         "pct_huber": float(100 * (g["policy_choice"] == "huber").mean()),
                         "pct_ridge": float(100 * (g["policy_choice"] == "ridge").mean())})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", nargs="+", default=["results/runs/urbino_*.csv"])
    p.add_argument("--out", default="results/checks/urbino")
    p.add_argument("--min-test-frac", type=float, default=0.5)
    p.add_argument("--calendar-only", action="store_true",
                   help="check 1 only, days split by calendar (weekday vs weekend/holiday) -> {out}_daytype_calendar.csv")
    a = p.parse_args()
    paths = sorted({f for pat in a.csv for f in glob.glob(pat)})
    print("files:", [Path(f).name for f in paths])
    df = summarize.recompute_policy(summarize.load(paths))
    kept, short = summarize.drop_short_splits(df, a.min_test_frac)
    if a.calendar_only:
        days = day_types(df, kept, calendar_only=True)
        by_day_type(kept, days).to_csv(f"{a.out}_daytype_calendar.csv", index=False)
        print("kept test days by type:", days.merge(kept[DAY].drop_duplicates(), on=DAY)
              .groupby(["horizon_min", "day_type"]).size().to_dict())
        print("holidays among kept days:", int(days.merge(kept[DAY].drop_duplicates(), on=DAY)["holiday"].sum()))
        return
    days = day_types(df, kept)

    dt = by_day_type(kept, days)
    dt.to_csv(f"{a.out}_daytype.csv", index=False)

    dropped = df[short].merge(days[DAY + ["day_type", "weekend"]], on=DAY)
    full = df.groupby(["device", "horizon_min"])["n_test"].max().rename("full_n_test")
    ddays = (dropped.groupby(DAY + ["day_type", "weekend", "n_test"]).size().rename("n_rows").reset_index()
             .join(full, on=["device", "horizon_min"]))
    ddays.to_csv(f"{a.out}_dropped.csv", index=False)

    pt = policy_threshold(df, a.min_test_frac)
    pt.to_csv(f"{a.out}_policy_threshold.csv", index=False)

    print(f"\nrows dropped: {int(short.sum())} on {len(ddays)} test days")
    print(dropped.groupby("day_type").size().rename("rows").to_frame()
          .join(ddays.groupby("day_type").size().rename("days")).to_string())
    print("\nn_test of dropped days (full day =", full.to_dict(), ")")
    print(ddays["n_test"].describe().round(1).to_string())
    print("\nkept test days by type:", days.merge(kept[DAY].drop_duplicates(), on=DAY)
          .groupby(["horizon_min", "day_type"]).size().to_dict())


if __name__ == "__main__":
    main()
