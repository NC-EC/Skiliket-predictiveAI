"""Adapter tests against the real downloaded files (skipped when a file is absent; run download_datasets.sh)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coldstart.adapters import dalton, ediaqi, urbino
from coldstart.schema import present_channels, validate

RAW = Path(__file__).parents[1] / "data/raw"
EDIAQI = RAW / "ediaqi"
DALTON = RAW / "dalton"
URBINO = RAW / "urbino"


def test_validate_accepts_absent_channel_but_not_partial_one():
    """Synthetic frame, code test only."""
    df = pd.DataFrame({"ts": pd.date_range("2025-01-01", periods=3, freq="1min"), "segment": 0, "valid": True,
                       "eco2": np.nan, "tvoc": np.nan, "aqi": np.nan, "temp_c": 20.0, "rh_pct": 40.0})
    validate(df)
    assert present_channels(df) == ["temp_c", "rh_pct"]
    df.loc[1, "rh_pct"] = np.nan
    with pytest.raises(ValueError, match="rh_pct"):
        validate(df)


@pytest.fixture(scope="module")
def all_devices():
    return ediaqi.load_all(EDIAQI)


@pytest.mark.skipif(not (EDIAQI / "Indoor_sensor_provider_1.csv").exists(), reason="real EDIAQI files not present")
class TestEdiaqiReal:
    def test_devices(self, all_devices):
        assert len(all_devices) == 22
        assert {"provider1", "provider4_s2", "provider5_s4", "provider6_s4", "provider6_s15", "reference"} <= set(all_devices)

    def test_provider1_profile(self, all_devices):
        df, meta, rep = all_devices["provider1"]
        assert rep["rows_raw"] == 28387
        assert rep["rows_usable"] == 28246
        assert rep["invalid_by_reason"]["dst_ambiguous_hour"] == 117
        assert rep["native_step_s"] == 60 and rep["max_gap_minutes"] == 5
        assert present_channels(df) == ["eco2", "tvoc", "temp_c", "rh_pct"]
        assert df["eco2"].between(440, 2600).all()
        assert df["tvoc"].between(0, 500).all()
        assert meta.timezone.startswith("Europe/Zagreb")

    def test_units_and_absent_tvoc(self, all_devices):
        # provider 4 reports TVOC in ppm; after x1000 it must sit in the same ppb range as the ppb providers
        assert 150 < all_devices["provider4_s1"][0]["tvoc"].median() < 400
        for d in ("provider2", "provider5_s1", "reference"):  # index or mg/m3: never mapped to ppb
            assert all_devices[d][0]["tvoc"].isna().all()
        for d, (df, _, _) in all_devices.items():
            assert df["aqi"].isna().all()
            assert df["temp_c"].between(10, 40).all(), d
            assert df["rh_pct"].between(20, 80).all(), d

    def test_dst_hour_dropped_and_no_duplicates(self, all_devices):
        for d, (df, _, _) in all_devices.items():
            t = df["ts"]
            assert not ((t >= "2024-10-27 02:00") & (t < "2024-10-27 03:00")).any(), d
            assert not t.duplicated().any(), d

    def test_coarse_devices_get_wider_gap(self, all_devices):
        assert all_devices["provider2"][2]["max_gap_minutes"] == 15
        assert all_devices["provider4_s1"][2]["rows_usable"] == 342112


@pytest.mark.skipif(not (DALTON / "Processed/H1").exists(), reason="real DALTON files not present (git lfs)")
class TestDaltonReal:
    def test_devices_skip_duplicate(self):
        devs = dalton.devices(DALTON)
        assert len(devs) == 68
        assert "F1_23" in devs and "F2_24" not in devs
        assert {"H1_41", "H2_11", "H4_11", "R1_51", "C1_25"} <= set(devs)  # ID 11 exists in H2 and H4

    def test_h1_41_profile(self):
        df, meta, rep = dalton.load(DALTON, "H1_41")
        assert rep["days"] == 26 and rep["rows_raw"] == 26 * 86400
        assert rep["rows_usable"] == 2044610
        assert rep["native_step_s"] == 1
        assert present_channels(df) == ["eco2", "tvoc", "temp_c", "rh_pct"]
        assert df["eco2"].between(250, 10000).all() and df["tvoc"].between(0, 1000).all()
        assert df["temp_c"].between(5, 45).all() and df["rh_pct"].between(5, 100).all()
        assert meta.timezone == "unknown"

    def test_c1_drops_clock_errors_and_smoothing_artifacts(self):
        df, _, rep = dalton.load(DALTON, "C1_25")
        assert rep["invalid_by_reason"]["clock_error"] == 2 * 86400  # the two 2028 folders
        assert df["ts"].max() < pd.Timestamp("2022-10-01")
        assert (df[["eco2", "temp_c", "rh_pct"]] > 0).all().all()
        assert df["temp_c"].min() >= 5

    def test_events(self):
        ev = dalton.load_events(DALTON, "H1")
        assert len(ev) == 1700 and ev["ts"].is_monotonic_increasing
        assert set(dalton.load_events(DALTON, "H12")["area"]) == {"Kitchen", "Bedroom"}


@pytest.mark.skipif(not (URBINO / "olivetti.csv").exists(), reason="real Urbino files not present")
def test_urbino_real_files_match_profile():
    res = urbino.load_all(URBINO)
    assert {d: r["rows_raw"] for d, (_, _, r) in res.items()} == {"olivetti": 91178, "turing": 87826, "von_neumann": 90111}
    assert res["turing"][2]["invalid_by_reason"] == {"eco2_sensor_ceiling": 35}
    for d, (df, meta, rep) in res.items():
        assert rep["native_step_s"] == 300, d
        assert present_channels(df) == ["eco2", "temp_c", "rh_pct"], d
        assert df["eco2"].between(400, 5000, inclusive="left").all(), d
        assert df["ts"].min() == pd.Timestamp("2024-01-01 00:05") and meta.timezone.startswith("UTC")
