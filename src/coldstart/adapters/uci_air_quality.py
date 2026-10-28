"""Adapter for the UCI Air Quality dataset (UCI id 360, research use; De Vito et al. 2008).

`path` is AirQualityUCI.csv from the UCI zip (';'-separated, ',' decimals, dd/mm/yyyy + HH.MM.SS).
One multisensor device at road level in an Italian city, hourly averages, Mar 2004 - Apr 2005.

Verified against the real file (see tests). Conventions learned from the file:
  * -200 marks a missing value (invalid, dropped). T, RH and AH are missing on the same 366 rows.
  * 114 trailing all-empty rows are CSV padding, not readings.
  * There is no eCO2, TVOC or AQI: only temperature and relative humidity map to canonical channels.
    NMHC(GT) (non-methane hydrocarbons, ug/m3, 90 % missing) and C6H6(GT) are NOT TVOC and are not mapped.
  * Outdoor deployment; the time zone (and whether DST was applied) is not stated and stays 'unknown'.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..schema import DatasetMeta, validate
from .base import build_report, default_max_gap, finalize, invalid_reasons, median_step_s

COLUMN_MAP = {"T": "temp_c", "RH": "rh_pct"}
MISSING = -200


def load(path: str | Path, max_gap_minutes: float | None = None) -> tuple[pd.DataFrame, DatasetMeta, dict]:
    raw = pd.read_csv(path, sep=";", decimal=",")
    raw = raw.loc[:, ~raw.columns.str.startswith("Unnamed")]
    missing = [c for c in ["Date", "Time", *COLUMN_MAP] if c not in raw.columns]
    if missing:
        raise ValueError(f"unexpected UCI Air Quality layout, missing columns: {missing}")
    raw = raw[raw["Date"].notna()].reset_index(drop=True)  # trailing padding rows

    df = raw[list(COLUMN_MAP)].rename(columns=COLUMN_MAP).mask(lambda d: d == MISSING)
    df.insert(0, "ts", pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H.%M.%S"))
    reasons = invalid_reasons(df, list(COLUMN_MAP.values()))
    valid = ~pd.concat(reasons, axis=1).any(axis=1)

    step_s = median_step_s(df["ts"])
    gap = default_max_gap(df["ts"]) if max_gap_minutes is None else max_gap_minutes
    out = finalize(df, valid, gap)
    validate(out)
    report = build_report(len(df), reasons, out, gap, step_s)

    meta = DatasetMeta(
        name="UCI Air Quality",
        source="UCI Machine Learning Repository, dataset 360 (air+quality.zip)",
        license="research use (per UCI page; verify before redistribution)",
        device_id="single multisensor device (not identified)",
        cadence_note="hourly averages",
        timezone="unknown",
        unit_assumptions={"temp_c": "degC (UCI variable description)", "rh_pct": "percent (UCI variable description)"},
        notes=["Outdoor, road-level deployment, not indoor",
               "No eCO2 / TVOC / AQI channels: usable only for covariates, not as a target",
               "NMHC(GT) and C6H6(GT) are not TVOC and are not mapped",
               "-200 treated as missing (invalid, dropped)"],
        verified_against_real_file=True,
    )
    return out, meta, report
