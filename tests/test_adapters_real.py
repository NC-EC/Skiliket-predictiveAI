"""Adapter tests against the real downloaded files (skipped when a file is absent; run download_datasets.sh)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coldstart.adapters import ediaqi, quant, uci_air_quality
from coldstart.schema import present_channels, validate

RAW = Path(__file__).parents[1] / "data/raw"
EDIAQI = RAW / "ediaqi"
UCI = RAW / "uci/AirQualityUCI.csv"
QUANT = RAW / "quant/QUANT_SensorSystems_hourly.csv"


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


@pytest.mark.skipif(not UCI.exists(), reason="real UCI Air Quality file not present")
def test_uci_real_file_matches_profile():
    df, meta, rep = uci_air_quality.load(UCI)
    assert rep["rows_raw"] == 9357
    assert rep["invalid_by_reason"] == {"temp_c_missing": 366, "rh_pct_missing": 366}
    assert rep["rows_usable"] == 8991
    assert rep["max_gap_minutes"] == 90
    assert present_channels(df) == ["temp_c", "rh_pct"]
    assert df["temp_c"].between(-2, 45).all() and df["rh_pct"].between(9, 89).all()
    assert meta.timezone == "unknown"


@pytest.mark.skipif(not QUANT.exists(), reason="real QUANT file not present")
def test_quant_real_file_matches_profile():
    res = quant.load_all(QUANT)
    assert len(res) == 68
    df, meta, rep = res["AQM389_1_London"]
    assert rep["rows_raw"] == rep["rows_usable"] == 20157
    df, meta, rep = res["Zep344_2_Manchester"]  # -81 / 331 degC readings must be gone
    assert rep["invalid_by_reason"]["temp_c_out_of_range"] == 386
    for d, (df, meta, _) in res.items():
        assert present_channels(df) == ["temp_c", "rh_pct"], d
        assert df["temp_c"].between(-40, 60).all() and (df["rh_pct"] > 0).all() and (df["rh_pct"] <= 100).all()
        assert meta.timezone == "unknown"
