"""Long-context inputs. SYNTHETIC grids only (code tests, not results)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from coldstart import evaluate as ev
from coldstart import longctx as lc


def _grid(days=6, step=10, gap=None):
    t = pd.date_range("2025-03-03", periods=days * 1440 // step, freq=f"{step}min")
    hour = t.hour + t.minute / 60
    y = 450 + 300 * ((hour >= 9) & (hour < 12))  # a daily 3-hour peak (SYNTHETIC)
    g = pd.DataFrame({"ts": t, "eco2": y.astype(float), "temp_c": 22.0})
    if gap is not None:
        g = g[(g.ts < gap[0]) | (g.ts >= gap[1])].reset_index(drop=True)
    g["segment"] = (g.ts.diff() > pd.Timedelta(minutes=step)).cumsum()
    return g


def test_hourly_means_relative_to_now_and_kit_start():
    g = _grid()
    ctx = lc.LongContext(g, "eco2", 10)
    t0 = np.array([np.datetime64("2025-03-05T13:00")])  # one hour after the peak ended
    y0 = np.array([450.0])
    cols = ctx.columns(t0, y0, t0 + np.timedelta64(30, "m"), np.datetime64("2025-03-03T00:00"))
    assert cols.shape == (1, len(lc.LONGCTX))
    # bins are right-closed: (12:00, 13:00] -> 450; (11:00, 12:00] -> five peak points and 12:00 (450);
    # (10:00, 11:00] and (9:00, 10:00] -> peak; (8:00, 9:00] -> only 9:00 is in the peak
    assert cols[0, 0] == 0 and np.allclose(cols[0, 1:4], [250, 300, 300]) and np.isclose(cols[0, 4], 50)
    assert cols[0, lc.HOURS] == 1.0 and cols[0, -1] == 1.0 and cols[0, -2] == 0.0  # flat at 12:00-12:30 yesterday
    # a kit switched on at 07:00 knows the bins (07:00, 13:00] plus its 07:00 reading in bin (06:00, 07:00], and has
    # no yesterday
    late = ctx.columns(t0, y0, t0 + np.timedelta64(30, "m"), np.datetime64("2025-03-05T07:00"))
    assert late[0, lc.HOURS] == 7 / 24 and late[0, -1] == 0 and (late[0, 7:lc.HOURS] == 0).all()


def test_gap_reduces_available_bins_without_filling():
    g = _grid(gap=("2025-03-05T02:00", "2025-03-05T05:00"))
    ctx = lc.LongContext(g, "eco2", 10)
    t0 = np.array([np.datetime64("2025-03-05T08:00")])
    cols = ctx.columns(t0, np.array([450.0]), t0 + np.timedelta64(30, "m"), np.datetime64("2025-03-03"))
    # readings 02:00-04:50 missing: bins (02:00, 03:00] and (03:00, 04:00] are empty; the bins around them keep their
    # remaining readings (no value is filled in)
    assert cols[0, lc.HOURS] == 22 / 24


def test_runner_long_context_end_to_end(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("rlc", Path(__file__).parents[1] / "scripts/run_learning_curve.py")
    rlc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rlc)
    g = _grid(days=10)
    monkeypatch.setattr(rlc, "load_grid", lambda *a, **k: (g, ["eco2", "temp_c"], "synthetic"))
    res = {}
    for flag in ("0", "24"):
        out = tmp_path / f"l{flag}.csv"
        monkeypatch.setattr(sys, "argv", ["x", "--dataset", "syn", "--path", ".", "--cadence", "10", "--horizon", "30",
                                          "--days", "1", "3", "--models", "ridge", "policy", "--seeds", "1",
                                          "--long-context", flag, "--out", str(out)])
        rlc.main()
        res[flag] = pd.read_csv(out)
    df = res["24"]
    assert (df["long_context_h"] == 24).all() and {"ridge", "policy"} <= set(df["model"])
    assert np.isfinite(df["rel_mae"]).all()
    # same splits and test windows; only the learned models' inputs differ
    key = ["train_days", "split", "model"]
    a, b = res["0"].set_index(key), df.set_index(key)
    assert (a.loc[(slice(None), slice(None), "persistence"), "mae"].values
            == b.loc[(slice(None), slice(None), "persistence"), "mae"].values).all()
    assert not np.allclose(a.loc[(slice(None), slice(None), "ridge"), "mae"], b.loc[(slice(None), slice(None), "ridge"), "mae"])
