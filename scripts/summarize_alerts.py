"""Summarize scripts/run_alerts.py CSVs.

Per device ({out}_devices.csv): over the kept test splits, recall = sum of hits / sum of events, false alarms per day =
sum of false alarms / sum of test-day spans, median lead time = median of the split medians. Short test splits are
dropped first with the same rule as summarize.py (--min-test-frac of the largest n_test per device and horizon).

Across devices ({out}_overall.csv), per (dataset, target, horizon, model, D, q, mode, budget): devices, devices with at
least one event, median and pooled recall, median FA/day, share of devices within the budget on the test days, median
lead time, and (with --events runs) recall on events that match an activity annotation.

  python scripts/summarize_alerts.py results/alerts/*.csv --out results/summary_alerts_dalton_eco2
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("summarize", Path(__file__).parent / "summarize.py")
_summ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_summ)

KEYS = ["dataset", "target", "horizon_min", "model", "train_days", "alert_q", "mode", "budget"]


def per_device(df: pd.DataFrame, min_test_frac: float = 0.5) -> pd.DataFrame:
    df, _ = _summ.drop_short_splits(df, min_test_frac)
    df = df.assign(budget=df["budget"].fillna(-1))  # raw rows have no budget; keep them in groupby
    agg = {"n_splits": ("split", "nunique"), "events": ("n_events", "sum"), "hits": ("hits", "sum"),
           "false_alarms": ("false_alarms", "sum"), "test_days": ("span_days", "sum"),
           "median_lead_min": ("median_lead_min", "median"), "alarm_frac": ("alarm_frac", "mean")}
    if "n_events_annotated" in df:
        agg |= {"events_annotated": ("n_events_annotated", "sum"), "hits_annotated": ("hits_annotated", "sum")}
    g = df.groupby(KEYS + ["device"]).agg(**agg).reset_index()
    g["recall"] = np.where(g["events"] > 0, g["hits"] / g["events"].where(g["events"] > 0), np.nan)
    g["fa_per_day"] = g["false_alarms"] / g["test_days"].where(g["test_days"] > 0)
    g["within_budget"] = np.where(g["budget"] >= 0, g["fa_per_day"] <= g["budget"], np.nan)
    g["budget"] = g["budget"].replace(-1, np.nan)
    return g


def overall(dev: pd.DataFrame) -> pd.DataFrame:
    d = dev.assign(budget=dev["budget"].fillna(-1))
    rows = []
    for k, g in d.groupby(KEYS):
        ev = g[g["events"] > 0]
        r = dict(zip(KEYS, k))
        r |= {"n_devices": len(g), "n_devices_with_events": len(ev),
              "median_recall": ev["recall"].median() if len(ev) else np.nan,
              "pooled_recall": g["hits"].sum() / g["events"].sum() if g["events"].sum() else np.nan,
              "median_fa_per_day": g["fa_per_day"].median(),
              "share_within_budget": g["within_budget"].astype(float).mean() if r["budget"] >= 0 else np.nan,
              "median_lead_min": g["median_lead_min"].median(), "median_alarm_frac": g["alarm_frac"].median()}
        if "events_annotated" in g and g["events_annotated"].sum():
            r |= {"events_annotated": int(g["events_annotated"].sum()),
                  "share_events_annotated": g["events_annotated"].sum() / g["events"].sum(),
                  "pooled_recall_annotated": g["hits_annotated"].sum() / g["events_annotated"].sum()}
        r["budget"] = np.nan if r["budget"] < 0 else r["budget"]
        rows.append(r)
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--min-test-frac", type=float, default=0.5)
    a = p.parse_args(argv)
    df = pd.concat([pd.read_csv(f) for f in a.csvs], ignore_index=True)
    dev = per_device(df, a.min_test_frac)
    ov = overall(dev)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    dev.to_csv(f"{a.out}_devices.csv", index=False)
    ov.to_csv(f"{a.out}_overall.csv", index=False)
    show = ov[(ov["mode"] == "tuned") & (ov["budget"] == 1) & (ov["alert_q"] == 90)]
    cols = ["horizon_min", "model", "train_days", "n_devices_with_events", "median_recall", "pooled_recall",
            "median_fa_per_day", "share_within_budget", "median_lead_min", "median_alarm_frac"]
    print(show[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
