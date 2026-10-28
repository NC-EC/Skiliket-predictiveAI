import numpy as np
import pandas as pd

from coldstart import cache


def _fake_load(calls):
    def load(path, device=None):
        """Synthetic 1-min series ONLY for code tests (not results)."""
        calls.append(device)
        ts = pd.date_range("2025-01-01", periods=600, freq="1min")
        df = pd.DataFrame({"ts": ts, "segment": 0, "eco2": 500 + np.arange(600.0), "tvoc": 100.0,
                           "temp_c": 22.0, "rh_pct": np.nan})
        return df, type("Meta", (), {"name": "fake"})(), {}
    return load


def test_grid_is_read_from_cache_on_second_call(tmp_path, monkeypatch):
    from coldstart.adapters import skiliket

    calls = []
    monkeypatch.setattr(skiliket, "load", _fake_load(calls))
    g1, chs1, name = cache.load_grid("skiliket", str(tmp_path), "dev", "eco2", 5, cache_dir=tmp_path / "c")
    g2, chs2, _ = cache.load_grid("skiliket", str(tmp_path), "dev", "eco2", 5, cache_dir=tmp_path / "c")
    assert calls == ["dev"]  # raw files read once
    assert name == "fake" and chs1 == chs2 == ["eco2", "tvoc", "temp_c"]  # rh_pct is all NaN: not a covariate
    pd.testing.assert_frame_equal(g1, g2)
    assert len(g1) == 120  # 600 min at 5 min

    cache.load_grid("skiliket", str(tmp_path), "dev", "tvoc", 5, cache_dir=tmp_path / "c")
    cache.load_grid("skiliket", str(tmp_path), "dev", "eco2", 5, cache_dir=None)
    assert calls == ["dev", "dev", "dev"]  # another target and a disabled cache both re-read
