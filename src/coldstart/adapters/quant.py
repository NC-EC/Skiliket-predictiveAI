"""Adapter for QUANT (Zenodo record 10775692, CC BY 4.0; Diez et al. 2024).

`path` is QUANT_SensorSystems_hourly.csv (long format: time, location, instrument, sensornumber, version,
measurand, measurement, flag, flagreason). 49 commercial sensor systems co-located with reference monitors
at three UK urban sites (London, Manchester, York), hourly, Dec 2019 - Oct 2022.

Verified against the real file (see tests). Conventions learned from the file:
  * Measurands are PM1/PM2.5/PM10, NO, NO2, O3, Temperature, RelHumidity, Pressure. There is NO CO2 and
    NO TVOC (the record description agrees), so only temperature and RH map to canonical channels.
  * Units move between sites; a device here is one deployment, '<instrument>_<sensornumber>_<location>'.
  * Temperature / RH exist only in the 'out-of-box' version; other versions are not read.
  * flag == 'Error' (e.g. "Temperature sensor fault") is invalid; 'Warning' ("Retrospectively collected")
    and 'Info' are kept.
  * Implausible values occur (e.g. -130 or 8020 degC, RH 0 %) and are dropped by the plausibility limits.
  * Outdoor deployment; the time zone is not stated in the record and stays 'unknown'.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..schema import DatasetMeta, validate
from .base import build_report, default_max_gap, finalize, invalid_reasons, median_step_s

MEASURANDS = {"Temperature": "temp_c", "RelHumidity": "rh_pct"}
VERSION = "out-of-box"
USECOLS = ["time", "location", "instrument", "sensornumber", "version", "measurand", "measurement", "flag"]


def _read(path: str | Path) -> pd.DataFrame:
    """Temperature / RH rows only, read in chunks (the full file is ~480 MB)."""
    parts = []
    for chunk in pd.read_csv(path, usecols=USECOLS, chunksize=2_000_000,
                             dtype={"sensornumber": "Int64", "flag": str, "measurand": str}):
        parts.append(chunk[chunk["measurand"].isin(list(MEASURANDS)) & (chunk["version"] == VERSION)])
    raw = pd.concat(parts, ignore_index=True)
    raw["device"] = raw["instrument"] + "_" + raw["sensornumber"].astype(str) + "_" + raw["location"]
    return raw


def devices(path: str | Path, _raw: pd.DataFrame | None = None) -> list[str]:
    raw = _read(path) if _raw is None else _raw
    return sorted(raw["device"].unique())


def load(path: str | Path, device: str, max_gap_minutes: float | None = None,
         _raw: pd.DataFrame | None = None) -> tuple[pd.DataFrame, DatasetMeta, dict]:
    raw = _read(path) if _raw is None else _raw
    sub = raw[raw["device"] == device]
    if sub.empty:
        raise ValueError(f"unknown QUANT device {device!r}; available: {devices(path, raw)}")

    wide = sub.pivot(index="time", columns="measurand", values=["measurement", "flag"])
    df = pd.DataFrame({"ts": pd.to_datetime(wide.index, format="%Y-%m-%d %H:%M:%S")})
    channels = []
    for measurand, ch in MEASURANDS.items():
        if ("measurement", measurand) in wide.columns:
            df[ch] = wide[("measurement", measurand)].to_numpy(dtype=float)
            channels.append(ch)
    flags = wide["flag"].to_numpy()
    reasons = invalid_reasons(df, channels)
    reasons["flag_error"] = pd.Series((flags == "Error").any(axis=1), index=df.index)
    valid = ~pd.concat(reasons, axis=1).any(axis=1)

    step_s = median_step_s(df["ts"])
    gap = default_max_gap(df["ts"]) if max_gap_minutes is None else max_gap_minutes
    out = finalize(df, valid, gap)
    validate(out)
    report = build_report(len(df), reasons, out, gap, step_s)

    instrument, sensornumber, location = sub.iloc[0][["instrument", "sensornumber", "location"]]
    meta = DatasetMeta(
        name=f"QUANT {device}",
        source="Zenodo record 10775692 (doi:10.5281/zenodo.10775692), QUANT_SensorSystems_hourly.csv",
        license="CC BY 4.0",
        device_id=device,
        cadence_note="hourly",
        timezone="unknown",
        unit_assumptions={"temp_c": "degC (inferred from values; not stated in the file)",
                          "rh_pct": "percent (inferred from values; not stated in the file)"},
        notes=[f"instrument {instrument}, sensor number {sensornumber}, site {location}",
               "Outdoor urban co-location site, not indoor",
               "No CO2 / TVOC / AQI channels: usable only for covariates, not as a target",
               f"version '{VERSION}' only"],
        verified_against_real_file=True,
    )
    return out, meta, report


def load_all(path: str | Path, max_gap_minutes: float | None = None) -> dict[str, tuple[pd.DataFrame, DatasetMeta, dict]]:
    """Every deployment, reading the large CSV once."""
    raw = _read(path)
    return {d: load(path, d, max_gap_minutes, _raw=raw) for d in devices(path, raw)}
