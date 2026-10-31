"""Weekly-pattern inputs. All data here is SYNTHETIC (code tests only, not results)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from coldstart import evaluate as ev
from coldstart import weekly as wk


def _grid(days=21, step=10):
    """A room used Mon-Fri 9-12 (CO2 +400 ppm), empty otherwise, on a regular grid, one segment."""
    t = pd.date_range("2024-03-04", periods=days * 1440 // step, freq=f"{step}min")  # 2024-03-04 is a Monday
    busy = (t.day_of_week < 5) & (t.hour >= 9) & (t.hour < 12)
    rng = np.random.default_rng(0)
    return pd.DataFrame({"ts": t, "segment": 0, "eco2": 450 + 400 * busy + rng.normal(0, 5, len(t)),
                         "temp_c": 22.0 + rng.normal(0, 0.1, len(t))})


def test_last_week_change_reads_the_same_interval_one_week_earlier():
    g = _grid()
    w = ev.make_windows(g, "eco2", ["eco2", "temp_c"], 6, 3)
    ch = wk.last_week_change(g, "eco2", w.t0, w.t_target)
    s = g.set_index("ts")["eco2"]
    i = np.where(w.t0 == np.datetime64("2024-03-12T08:50"))[0][0]  # Tuesday of week 2, just before class
    expect = s[pd.Timestamp("2024-03-05T09:20")] - s[pd.Timestamp("2024-03-05T08:50")]
    assert np.isclose(ch[i], expect) and ch[i] > 300  # last Tuesday the class started in this interval
    assert np.isnan(ch[: 6 * 24 * 7 - 10]).all()  # first week: nothing one week earlier


def test_weekly_columns_respect_the_kit_start():
    g = _grid()
    w = ev.make_windows(g, "eco2", ["eco2", "temp_c"], 6, 3)
    ch = wk.last_week_change(g, "eco2", w.t0, w.t_target)
    late_start = np.datetime64("2024-03-15")  # a kit started on day 11 has no last-week values before day 18
    cols = wk.weekly_columns(ch, w.t0, late_start)
    assert cols.shape == (len(w.y), len(wk.WEEKLY))
    before = w.t0 < late_start + wk.WEEK
    assert (cols[before, 1] == 0).all() and (cols[before, 0] == 0).all()
    assert cols[~before, 1].mean() > 0.9
    assert (cols[:, 2:].sum(axis=1) == 1).all()  # one-hot day of week
    assert np.array_equal(wk.weekly_naive(w.y0, cols)[before], w.y0[before])


def _runner():
    spec = importlib.util.spec_from_file_location("rlc", Path(__file__).parents[1] / "scripts/run_learning_curve.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runner_weekly_flag_end_to_end(tmp_path, monkeypatch):
    rlc = _runner()
    g = _grid(days=24)
    monkeypatch.setattr(rlc, "load_grid", lambda *a, **k: (g, ["eco2", "temp_c"], "synthetic"))
    out = tmp_path / "w.csv"
    monkeypatch.setattr(sys, "argv", ["x", "--dataset", "syn", "--path", ".", "--cadence", "10", "--horizon", "30",
                                      "--days", "3", "14", "--models", "ridge", "huber", "policy", "--seeds", "1",
                                      "--weekly", "--out", str(out)])
    rlc.main()
    df = pd.read_csv(out)
    assert df["weekly"].all() and "weekly_naive" in set(df["model"])
    # with a strict weekly timetable, last week's change is the right forecast once the kit has a week of history
    d14 = df[(df.train_days == 14)].groupby("model")["rel_mae"].median()
    assert d14["weekly_naive"] < 0.8 and d14["huber"] < 1.0
    # with D = 3 no window has last week's value, so weekly_naive is exactly persistence
    d3 = df[(df.train_days == 3) & (df.model == "weekly_naive")]
    assert np.allclose(d3["rel_mae"], 1.0)
