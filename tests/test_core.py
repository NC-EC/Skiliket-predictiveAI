from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coldstart import evaluate as ev
from coldstart.adapters import skiliket
from coldstart.adapters.base import finalize
from coldstart.cost import measure_fit
from coldstart.schema import validate

REAL = Path(__file__).parents[1] / "data/raw/skiliket_zenodo_17398551.xlsx"


def synth(n=600, gap_at=None):
    """Synthetic series ONLY for code tests (not results)."""
    ts = pd.date_range("2025-01-01", periods=n, freq="1min")
    if gap_at:
        ts = ts.where(np.arange(n) < gap_at, ts + pd.Timedelta(minutes=30))
    rng = np.random.default_rng(0)
    eco2 = 500 + np.cumsum(rng.normal(0, 5, n))
    return pd.DataFrame({"ts": ts, "eco2": eco2, "tvoc": 100.0, "aqi": 2, "temp_c": 22.0, "rh_pct": 40.0})


def test_finalize_segments_and_drops_invalid():
    df = synth(100, gap_at=50)
    valid = pd.Series(True, index=df.index)
    valid[10] = False
    out = finalize(df, valid, max_gap_minutes=5)
    validate(out)
    assert len(out) == 99
    assert out["segment"].nunique() == 2


def test_mase_is_one_for_persistence_on_train_like_data():
    y0 = np.array([1.0, 2, 3, 4])
    y = np.array([2.0, 3, 4, 5])
    assert ev.mase(y, y0, y, y0) == pytest.approx(1.0)
    assert ev.mase(y, y, y, y0) == pytest.approx(0.0)


def test_windows_do_not_cross_segments():
    df = synth(100, gap_at=50)
    df["segment"] = (df["ts"].diff().dt.total_seconds().div(60) > 5).cumsum()
    w = ev.make_windows(df, "eco2", ["eco2"], n_lags=5, horizon_steps=10)
    span = (w.t_target - w.t0).astype("timedelta64[m]").astype(int)
    assert (span == 10).all()  # target is always exactly 10 min after origin => no crossing the 30-min gap


def test_rolling_origin_no_leakage():
    df = synth(3000)
    df["segment"] = 0
    w = ev.make_windows(df, "eco2", ["eco2"], n_lags=10, horizon_steps=15)
    purge = 15 + 10
    n = 0
    for tr, te in ev.rolling_origin(w, train_days=0.5, test_days=0.25, purge_minutes=purge):
        assert w.t_target[tr].max() < w.t0[te].min()          # no target of train overlaps test origins
        assert (w.t0[te].min() - w.t_target[tr].max()) >= np.timedelta64(purge - 15, "m")
        n += 1
    assert n >= 2


def test_baselines_and_cost():
    df = synth(1000)
    df["segment"] = 0
    w = ev.make_windows(df, "eco2", ["eco2"], n_lags=5, horizon_steps=5)
    idx = np.arange(len(w.y))
    assert np.allclose(ev.persistence(w, idx), w.y0)
    ma = ev.moving_average(w, idx, n_channels=1)
    assert np.allclose(ma, w.X.mean(axis=1))
    _, cost = measure_fit(lambda: sum(range(10000)))
    assert cost.wall_s >= 0 and cost.max_rss_mb > 0


@pytest.mark.skipif(not REAL.exists(), reason="real SKILIKET file not present")
def test_skiliket_real_file_matches_profile():
    df, meta, rep = skiliket.load(REAL)
    assert rep["rows_raw"] == 8897
    assert rep["rows_invalid_dropped"] == 6
    assert rep["rows_usable"] == 8891
    assert df["aqi"].between(1, 5).all()
    assert (df["eco2"] >= 400).all()
    assert meta.timezone == "unknown"
    assert 6.7 < rep["span_days"] < 6.8
