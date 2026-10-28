"""Adapter for the Urbino university classrooms dataset (Zenodo record 17487080, CC BY 4.0).

`path` is the folder holding olivetti.csv, turing.csv and von_neumann.csv. Three classrooms at the University
of Urbino (Italy), calendar year 2024. A device is one room: `load(path, device)` / `load_all(path)`.

Verified against the real files (see tests). Conventions learned from the files:
  * Time stamps are ISO 8601 with a 'Z' suffix, i.e. UTC; they are converted to naive UTC.
  * The record says 10-min sampling, but the files step every 5 min, with ~8 % single missing readings (10-min steps)
    and a few multi-day outages.
  * CO2 never drops below ~444 ppm and tops out at exactly 5000 ppm in one room: 5000 is the sensor ceiling
    (saturated, true value unknown) and is invalid.
  * Units are not stated anywhere in the record; ppm / degC / % are inferred from the value ranges.
  * There is no TVOC or AQI. PM and noise are out of scope.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..schema import DatasetMeta, validate
from .base import build_report, default_max_gap, finalize, invalid_reasons, median_step_s

ROOMS = ["olivetti", "turing", "von_neumann"]
COLUMN_MAP = {"timestamp": "ts", "co2": "eco2", "temperature": "temp_c", "humidity": "rh_pct"}
CO2_CEILING = 5000.0


def devices(path: str | Path) -> list[str]:
    return [r for r in ROOMS if (Path(path) / f"{r}.csv").exists()]


def load(path: str | Path, device: str, max_gap_minutes: float | None = None) -> tuple[pd.DataFrame, DatasetMeta, dict]:
    if device not in ROOMS:
        raise ValueError(f"unknown Urbino room {device!r}; available: {ROOMS}")
    raw = pd.read_csv(Path(path) / f"{device}.csv")
    missing = [c for c in COLUMN_MAP if c not in raw.columns]
    if missing:
        raise ValueError(f"unexpected Urbino layout, missing columns: {missing}")
    df = raw.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())]
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)

    channels = ["eco2", "temp_c", "rh_pct"]
    reasons = invalid_reasons(df, channels)
    reasons["eco2_sensor_ceiling"] = df["eco2"] >= CO2_CEILING
    valid = ~pd.concat(reasons, axis=1).any(axis=1)

    step_s = median_step_s(df["ts"])
    gap = default_max_gap(df["ts"]) if max_gap_minutes is None else max_gap_minutes
    out = finalize(df, valid, gap)
    validate(out)
    report = build_report(len(df), reasons, out, gap, step_s)

    meta = DatasetMeta(
        name=f"Urbino classroom {device}",
        source="Zenodo record 17487080 (doi:10.5281/zenodo.17487080)",
        license="CC BY 4.0",
        device_id=device,
        cadence_note=f"median native step {step_s:.0f} s (record states 10 min)",
        timezone="UTC (ISO 8601 'Z' suffix in the files)",
        unit_assumptions={"eco2": "ppm (inferred; sensor type not stated)", "temp_c": "degC (inferred)",
                          "rh_pct": "percent (inferred)"},
        notes=["University classroom, occupancy-driven CO2", f"CO2 >= {CO2_CEILING:g} ppm treated as sensor saturation",
               "No TVOC / AQI channels", "PM and noise dropped (out of scope)"],
        verified_against_real_file=True,
    )
    return out, meta, report


def load_all(path: str | Path, max_gap_minutes: float | None = None) -> dict[str, tuple[pd.DataFrame, DatasetMeta, dict]]:
    return {d: load(path, d, max_gap_minutes) for d in devices(path)}
