"""Adapter for DALTON (github.com/prasenjit52282/dalton-dataset, AGPL-3.0; cite arXiv 2407.14501).

`path` is the cloned repository root (CSVs come from Git LFS). 30 indoor sites in India (homes H, studios A,
labs R, canteens F, classrooms C), 69 devices, 2023-2024, one file per device and day:
`Processed/<Site>/<YYYY_MM_DD>/<ID>_<Loc>.csv`, 86,400 rows at 1 Hz. A device is '<Site>_<ID>' (IDs repeat across
sites). Load one with `load(path, device)`, all with `load_all(path)`, and activity labels with `load_events`.

Verified against the real files (see tests). The authors' preprocessing (library/preprocess.py) shapes what is usable:
  * Gaps up to 15 min were filled with the NEAREST reading and then marked Valid=1; longer gaps and the first
    59 s of each day (rolling-window warm-up) were set to 0, sometimes still with Valid=1.
  * Every channel was smoothed with a trailing 60 s rolling mean (causal, so no look-ahead).
  * CO2 > 10000 ppm was treated as an electrical-surge fault (Valid_CO2=0).
  * Device clocks were sometimes unset or wrong: folders dated 1970-01-01 and 2028 exist.
  We drop: Valid == 0, Valid_CO2 == 0, rows where CO2, T or RH is exactly 0 (zero-fill) and the 60 s after them
  (the rolling mean ramps up from the zeros), runs where T, RH, CO2 and VoC are all frozen for more than 60 s (the
  detectable part of the nearest-fill), time stamps outside 2022-2024, the CO2 error code 8191, and deep dips left by
  raw zeros shorter than the smoothing window (CO2 < 250 ppm, T < 5 degC, RH < 5 %). Shorter fills and shallow dips
  are smoothed away and cannot be detected: a documented deviation from the no-imputation rule (~1-2 % of rows).
  * F2_24 repeats F1_23 reading for reading on all 10 days, so devices() lists it once (as F1_23).
  * Units per the README: CO2 ppm, VoC ppb, T degC, H %. Time zone not stated (ESP32 clock): 'unknown'.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..schema import DatasetMeta, validate
from .base import build_report, default_max_gap, finalize, invalid_reasons, median_step_s

COLUMN_MAP = {"ts": "ts", "CO2": "eco2", "VoC": "tvoc", "T": "temp_c", "H": "rh_pct"}
FLAGS = ["Valid", "Valid_CO2"]
FROZEN_S = 60  # longer all-channel freezes are nearest-filled gaps, not readings
SMOOTH_S = 60  # authors' trailing rolling-mean window: readings this soon after a zero are mixed with zeros
CAMPAIGN = (pd.Timestamp("2022-01-01"), pd.Timestamp("2025-01-01"))  # 1970 (unset clock) and 2028 folders exist
# Raw zeros shorter than the smoothing window never reach exactly 0 after smoothing; they show up as deep dips.
# Floors below anything physical indoors catch the deep ones (shallow dips stay undetectable).
FLOORS = {"eco2": 250.0, "temp_c": 5.0, "rh_pct": 5.0}
CO2_ERROR_CODE = 8191.0  # 2**13 - 1, a sensor error value seen in H13_96
DUPLICATES = {"F2_24": "F1_23"}  # identical readings on all 10 days; listed once in devices()


def _files(path: Path, site: str, device_id: str) -> list[Path]:
    return sorted((path / "Processed" / site).glob(f"*/{device_id}_*.csv"))


def devices(path: str | Path) -> list[str]:
    found = {f"{f.parent.parent.name}_{f.name.split('_', 1)[0]}" for f in Path(path, "Processed").glob("*/*/*.csv")}
    return sorted(found - set(DUPLICATES))


def _frozen_runs(df: pd.DataFrame, channels: list[str], min_len_s: int) -> pd.Series:
    """True on every row of a run where all channels repeat the previous value for more than min_len_s rows."""
    same = df[channels].diff().abs().sum(axis=1).eq(0)
    run_id = (~same).cumsum()
    run_len = same.groupby(run_id).transform("sum")
    frozen = same & (run_len > min_len_s)
    # the first row of a run (the last real reading before the freeze) is kept
    return frozen


def load(path: str | Path, device: str, max_gap_minutes: float | None = None) -> tuple[pd.DataFrame, DatasetMeta, dict]:
    path = Path(path)
    site, _, device_id = device.partition("_")
    files = _files(path, site, device_id)
    if not files:
        raise ValueError(f"unknown DALTON device {device!r}; see devices(path)")
    raw = pd.concat([pd.read_csv(f, usecols=[*COLUMN_MAP, *FLAGS, "Loc"], dtype={"Loc": str}) for f in files],
                    ignore_index=True)
    raw["ts"] = pd.to_datetime(raw["ts"], format="%Y-%m-%d %H:%M:%S")
    raw = raw.sort_values("ts", kind="stable").reset_index(drop=True)  # flags stay aligned with readings
    locations = sorted(raw["Loc"].astype(str).unique())
    df = raw.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())]

    channels = ["eco2", "tvoc", "temp_c", "rh_pct"]
    reasons = invalid_reasons(df, channels)
    reasons["author_invalid"] = raw["Valid"].to_numpy() == 0
    reasons["author_invalid_co2"] = raw["Valid_CO2"].to_numpy() == 0
    zero = (df[["eco2", "temp_c", "rh_pct"]] == 0).any(axis=1)
    reasons["zero_filled"] = zero
    # rows are contiguous 1 Hz within each day, so a row count is a duration in seconds
    reasons["zero_smoothing_ramp"] = zero.astype(int).rolling(SMOOTH_S + 1, min_periods=1).max().astype(bool) & ~zero
    reasons["frozen_fill"] = _frozen_runs(df, channels, FROZEN_S)
    reasons["clock_error"] = (df["ts"] < CAMPAIGN[0]) | (df["ts"] >= CAMPAIGN[1])
    reasons["smoothed_dip"] = ~zero & pd.concat([df[c] < v for c, v in FLOORS.items()], axis=1).any(axis=1)
    reasons["eco2_error_code"] = df["eco2"] == CO2_ERROR_CODE
    reasons = {k: pd.Series(v, index=df.index) for k, v in reasons.items()}
    valid = ~pd.concat(reasons, axis=1).any(axis=1)

    step_s = median_step_s(df["ts"])
    gap = default_max_gap(df["ts"]) if max_gap_minutes is None else max_gap_minutes
    out = finalize(df, valid, gap)
    validate(out)
    report = build_report(len(df), reasons, out, gap, step_s)
    report["days"] = len(files)

    meta = DatasetMeta(
        name=f"DALTON {device}",
        source="github.com/prasenjit52282/dalton-dataset, Processed/ (Git LFS)",
        license="AGPL-3.0 (authors: free for non-commercial research); cite arXiv 2407.14501",
        device_id=device,
        cadence_note="1 Hz, authors' trailing 60 s rolling mean",
        timezone="unknown",
        unit_assumptions={"eco2": "ppm (README)", "tvoc": "ppb (README)", "temp_c": "degC (README)",
                          "rh_pct": "percent (README)"},
        notes=[f"site {site}, device {device_id}, location(s): {', '.join(locations)}",
               "Authors nearest-filled gaps <= 15 min and marked them valid; only frozen runs > 60 s are detectable "
               "and dropped, shorter fills remain (deviation from the no-imputation rule)",
               "All channels smoothed by the authors with a trailing 60 s mean",
               "Author zero-fill rows (gaps > 15 min, daily warm-up) and the 60 s smoothing ramps after them dropped",
               "Deep smoothing dips dropped by floors (CO2 250 ppm, T 5 degC, RH 5 %); shallow dips undetectable",
               *([f"Readings identical to {DUPLICATES[device]} (duplicate upload)"] if device in DUPLICATES else []),
               "No AQI channel"],
        verified_against_real_file=True,
    )
    return out, meta, report


def load_all(path: str | Path, max_gap_minutes: float | None = None) -> dict[str, tuple[pd.DataFrame, DatasetMeta, dict]]:
    return {d: load(path, d, max_gap_minutes) for d in devices(path)}


def load_events(path: str | Path, site: str) -> pd.DataFrame:
    """Activity annotations for a site (Metadata/Annotations_cleaned.csv): columns ts, label, area, participant.

    Some sites are annotated per area ('H12-Kitchen', 'H2-Floor0'); `area` keeps that suffix ('' otherwise).
    Labels are free text with names replaced by '_NAME_'.
    """
    ann = pd.read_csv(Path(path) / "Metadata" / "Annotations_cleaned.csv")
    site_part = ann["Site"].astype(str).str.split("-", n=1)
    sel = site_part.str[0] == site
    out = pd.DataFrame({
        "ts": pd.to_datetime(ann.loc[sel, "ts"], format="%Y-%m-%d %H:%M:%S"),
        "label": ann.loc[sel, "Label"].astype(str).str.strip(),
        "area": site_part[sel].str[1].astype(object).fillna(""),
        "participant": ann.loc[sel, "Customer"],
    })
    return out.sort_values("ts").reset_index(drop=True)
