import numpy as np
import pytest

from coldstart import alerts, stats
from coldstart.models import REGISTRY


def _data(n=800, lags=6, C=2, seed=0):
    """Synthetic AR data ONLY for code tests (not results)."""
    rng = np.random.default_rng(seed)
    s = np.zeros(n + lags + 3)
    for i in range(1, len(s)):
        s[i] = 0.9 * s[i - 1] + rng.normal()
    X = np.stack([np.stack([s[i : i + lags], np.zeros(lags)], axis=1).ravel() for i in range(n)])
    y0 = s[lags - 1 : lags - 1 + n]
    y = s[lags + 2 : lags + 2 + n]
    return X, y, y0


@pytest.mark.parametrize("name", list(REGISTRY))
def test_models_fit_predict_and_beat_zero_on_ar(name):
    X, y, y0 = _data()
    kw = {"n_channels": 2, "epochs": 15} if name == "gru" else {}
    m = REGISTRY[name](seed=0, **kw).fit(X[:600], y[:600], y0[:600])
    yp = m.predict(X[600:], y0[600:])
    assert yp.shape == (200,)
    assert np.isfinite(yp).all()
    # mean-reverting AR(0.9): learned models should beat persistence on average
    assert np.mean(np.abs(y[600:] - yp)) < 1.05 * np.mean(np.abs(y[600:] - y0[600:]))


def test_gru_is_small():
    X, y, y0 = _data(200)
    m = REGISTRY["gru"](seed=0, n_channels=2, epochs=2).fit(X, y, y0)
    assert m.n_params() < 2000


def test_alerts_perfect_forecast_hits_event_with_lead():
    t = np.arange("2025-01-01T00:00", "2025-01-01T06:00", np.timedelta64(5, "m"), dtype="datetime64[m]")
    y = np.zeros(len(t))
    y[40:45] = 10  # one event
    h = 15  # 3 steps
    t0, tt = t[:-3], t[3:]
    yt = y[3:]
    r = alerts.evaluate_alerts(t0, tt, yt, yt.copy(), threshold=5, horizon_min=h)
    assert r["n_events"] == 1 and r["hits"] == 1 and r["false_alarms"] == 0
    assert r["median_lead_min"] == 15


def test_alerts_false_alarm_counted():
    t = np.arange("2025-01-01T00:00", "2025-01-01T06:00", np.timedelta64(5, "m"), dtype="datetime64[m]")
    yt = np.zeros(len(t))
    yp = yt.copy()
    yp[10] = 10
    r = alerts.evaluate_alerts(t, t + np.timedelta64(15, "m"), yt, yp, threshold=5, horizon_min=15)
    assert r["n_events"] == 0 and r["false_alarms"] == 1


def test_stats_ci_and_dm():
    m, lo, hi = stats.bootstrap_ci(np.full(20, 0.8))
    assert lo == hi == m == pytest.approx(0.8)
    assert stats.meets_cold_start(np.random.default_rng(0).normal(0.8, 0.05, 50))
    assert not stats.meets_cold_start(np.random.default_rng(0).normal(0.99, 0.2, 10))
    rng = np.random.default_rng(1)
    e = rng.normal(size=500)
    dm, p = stats.diebold_mariano(e * 0.5, e)
    assert dm < 0 and p < 0.01
