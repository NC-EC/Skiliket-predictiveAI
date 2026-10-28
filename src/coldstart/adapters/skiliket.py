"""Adapter for the SKILIKET IoT environmental dataset (Zenodo, xlsx).

Verified against the real file (see tests). Conventions learned from the file:
  * 0 in eCO2, RH, temperature or AQI means an invalid reading (sensor warm-up/failed read).
  * AQI is an integer 1-5; eCO2 has a floor of 400 ppm.
  * Soil moisture is outside the scope of this study and is dropped.
  * Time stamps are naive; the time zone is NOT stated and stays 'unknown'.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..schema import DatasetMeta, validate
from .base import finalize

COLUMN_MAP = {
    "Time Stamp / Category": "ts",
    "AQI": "aqi",
    "CO2_e": "eco2",
    "Relative Humidity": "rh_pct",
    "TVOC": "tvoc",
    "Ambient Temperature": "temp_c",
}


def load(path: str | Path, max_gap_minutes: float = 5.0) -> tuple[pd.DataFrame, DatasetMeta, dict]:
    raw = pd.read_excel(path)
    missing = [c for c in COLUMN_MAP if c not in raw.columns]
    if missing:
        raise ValueError(f"unexpected SKILIKET layout, missing columns: {missing}")
    df = raw.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())]
    df["ts"] = pd.to_datetime(df["ts"])

    valid = (df.eco2 > 0) & (df.rh_pct > 0) & (df.temp_c > 0) & (df.aqi > 0)
    out = finalize(df, valid, max_gap_minutes)
    validate(out)

    report = {
        "rows_raw": int(len(raw)),
        "rows_invalid_dropped": int((~valid).sum()),
        "rows_usable": int(len(out)),
        "segments": int(out["segment"].nunique()),
        "start": out["ts"].min(),
        "end": out["ts"].max(),
        "span_days": float((out["ts"].max() - out["ts"].min()).total_seconds() / 86400),
    }
    meta = DatasetMeta(
        name="SKILIKET IoT environmental dataset",
        source="Zenodo (SKILIKET), file supplied by the team",
        license="CC BY 4.0 (per Zenodo record; verify on record page)",
        device_id="unknown (single device, not identified in file)",
        cadence_note="~1 reading/min nominal, irregular",
        timezone="unknown",
        unit_assumptions={
            "eco2": "ppm (inferred)",
            "tvoc": "ppb (inferred)",
            "temp_c": "degC (inferred)",
            "rh_pct": "percent (inferred)",
            "aqi": "1-5 (observed values)",
        },
        notes=["Soil moisture dropped (out of scope)", "Sensor model not stated; ENS160-like behaviour is an inference"],
        verified_against_real_file=True,
    )
    return out, meta, report
