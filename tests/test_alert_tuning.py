"""Alert operating point, annotation matching, run_alerts and summarize_alerts. All data here is SYNTHETIC (code
tests only, not results)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coldstart import alerts as al


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _times(n, step=5):
    return np.datetime64("2025-01-01T00:00") + np.arange(n) * np.timedelta64(step, "m")


def test_event_hit_flags_match_hits():
    t = _times(200)
    y = np.zeros(200)
    y[50:55] = 10
    y[150:155] = 10
    yp = np.zeros(200)
    yp[50] = 10  # forecast for the first onset, issued 15 min before it
    r = al.evaluate_alerts(t - np.timedelta64(15, "m"), t, y, yp, 5, 15, return_events=True)
    assert r["hits"] == 1 and list(r["event_hit"]) == [True, False] and len(r["event_onsets"]) == 2
    assert "event_hit" not in al.evaluate_alerts(t, t, y, yp, 5, 15)  # runner output unchanged


def test_choose_offset_keeps_budget_and_prefers_recall():
    rng = np.random.default_rng(0)
    t = _times(288 * 3)  # 3 days at 5 min
    y = rng.normal(0, 1, len(t))
    y[100:110] = y[400:410] = y[700:710] = 8
    yp = y.copy()  # target-time forecast from 15 min earlier: perfect
    t0 = t - np.timedelta64(15, "m")
    offs = al.offset_grid(rng.normal(0, 2, 500))
    off, r = al.choose_offset(t0, t, y, yp, 5, 15, budget_per_day=1, offsets=offs)
    assert r["fa_per_day"] <= 1 and r["recall"] == 1
    # an impossible budget falls back to the fewest false alarms
    noisy = yp + rng.normal(0, 3, len(t))
    off2, r2 = al.choose_offset(t0, t, y, noisy, 5, 15, budget_per_day=-1, offsets=offs)
    assert r2["fa_per_day"] == min(al.evaluate_alerts(t0, t, y, noisy, 5, 15, offset=o)["fa_per_day"] for o in offs)


def test_choose_offset_without_events_picks_most_sensitive_within_budget():
    t = _times(288)
    y = np.zeros(288)
    # forecast + offset > 5 alarms: 10 alarms all day (one merged alarm, > 1 per day over a 0.997-day span)
    off, r = al.choose_offset(t, t, y, y, 5, 15, budget_per_day=1, offsets=np.array([10.0, 4.0, 0.0, -2.0]))
    assert off == 4.0 and np.isnan(r["recall"])  # the most sensitive offset that keeps the budget


def test_annotation_shift_scan_finds_clock_offset():
    rng = np.random.default_rng(1)
    onsets = np.sort(np.datetime64("2025-01-01") + rng.integers(0, 30 * 1440, 80) * np.timedelta64(1, "m"))
    ann = onsets - np.timedelta64(20, "m") - np.timedelta64(330, "m")  # activity 20 min before, clock 5.5 h behind
    shifts = np.arange(-720, 721, 15)
    share = al.shift_scan(onsets, ann, shifts)
    # window [onset - 60, onset + 15] holds the annotation for shifts 290..365 min
    assert 290 <= shifts[np.argmax(share)] <= 365 and share[shifts == 330][0] == 1
    assert al.annotated(onsets, ann).mean() < 0.2


def test_head_val_split_is_time_ordered_with_gap():
    ra = _load("run_alerts")
    t0 = _times(100)[::-1]  # deliberately unsorted indices
    head, val = ra.head_val(np.arange(100), t0, 0.3, gap=9)
    assert len(val) == 30 and len(head) == 100 - 30 - 9
    assert t0[head].max() < t0[val].min()


@pytest.fixture
def synth_grid():
    """Three weeks of a daily CO2-like cycle with noise on a 5-min grid, one segment."""
    rng = np.random.default_rng(2)
    t = pd.date_range("2025-01-01", periods=288 * 21, freq="5min")
    hour = t.hour.to_numpy() + t.minute.to_numpy() / 60
    eco2 = 450 + 400 * np.exp(-((hour - 19) ** 2) / 4) + rng.normal(0, 20, len(t))
    g = pd.DataFrame({"ts": t, "segment": 0, "eco2": eco2, "temp_c": 22 + rng.normal(0, 0.2, len(t)),
                      "rh_pct": 50 + rng.normal(0, 1, len(t))})
    return g, ["eco2", "temp_c", "rh_pct"]


def test_run_alerts_end_to_end(tmp_path, monkeypatch, synth_grid):
    ra = _load("run_alerts")
    monkeypatch.setattr(ra, "load_grid", lambda *a, **k: (*synth_grid, "synthetic"))
    out = tmp_path / "a.csv"
    ra.main(["--dataset", "synthetic", "--path", ".", "--devices", "X1", "--horizon", "30", "--days", "3", "7",
             "--models", "persistence", "ridge", "policy", "--alert-q", "90", "--budgets", "1", "--out", str(out)])
    df = pd.read_csv(out)
    assert set(df["mode"]) == {"raw", "tuned"} and set(df["model"]) == {"persistence", "ridge", "policy"}
    starts = {D: set(g["test_start"]) for D, g in df.groupby("train_days")}
    assert starts[3] == starts[7]  # common test days for every D
    tuned = df[df["mode"] == "tuned"]
    assert tuned["val_fa_per_day"].notna().all() and (tuned["n_head"] + tuned["n_val"] < tuned["n_train"]).all()
    # the daily evening peak is predictable from the hour: a learned model should warn on most events
    assert tuned[tuned["model"] == "ridge"]["recall"].mean() > 0.5

    sa = _load("summarize_alerts")
    dev = sa.per_device(df)
    ov = sa.overall(dev)
    r = ov[(ov["mode"] == "tuned") & (ov["model"] == "ridge") & (ov["train_days"] == 7)].iloc[0]
    d = dev[(dev["mode"] == "tuned") & (dev["model"] == "ridge") & (dev["train_days"] == 7)].iloc[0]
    sub = df[(df["mode"] == "tuned") & (df["model"] == "ridge") & (df["train_days"] == 7)]
    assert d["recall"] == pytest.approx(sub["hits"].sum() / sub["n_events"].sum())
    assert d["fa_per_day"] == pytest.approx(sub["false_alarms"].sum() / sub["span_days"].sum())
    assert r["n_devices"] == 1 and r["budget"] == 1
    assert ov[ov["mode"] == "raw"]["budget"].isna().all()
