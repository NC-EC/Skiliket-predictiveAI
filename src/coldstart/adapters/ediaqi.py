"""Adapter for the EDIAQI indoor dataset (Zenodo record 14763137, CC BY 4.0).

`path` is the folder holding the CSVs extracted from the record's data.zip. One house in Zagreb,
11-31 Oct 2024, six low-cost sensor providers (some with several units) plus one reference device.
Each physical unit is a device; load one with `load(path, device)` or all with `load_all(path)`.

Verified against the real files (see tests). Conventions learned from the files:
  * Time stamps are local Zagreb wall clock (metadata: "Zagreb timezone (CET)"). Several devices repeat the
    hour 2024-10-27 02:00-03:00 when DST ends; that hour is ambiguous and is dropped on every device.
  * TVOC units differ by provider: ppb (1, 3, 6), ppm (4, converted x1000), mg/m3 (5, reference) and a
    unitless VOC index (2, 0-500, Sensirion-style despite "ppb" in the metadata). mg/m3 cannot become ppb
    without an assumed molar mass, so TVOC is left absent for providers 2, 5 and the reference.
  * CO2 is reported as CO2 in ppm; whether it is NDIR or MOx-derived eCO2 is not stated.
  * Wide multi-unit files (providers 4, 5, 6) hold one block of columns per unit; rows where a unit has no
    data at all are not that unit's readings and are ignored, rows with a partial reading are invalid.
  * People / door / motion logs exist but there is no labeled-events JSON in this record version.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..schema import DatasetMeta, validate
from .base import build_report, default_max_gap, finalize, invalid_reasons, median_step_s

DST_AMBIGUOUS = (pd.Timestamp("2024-10-27 02:00"), pd.Timestamp("2024-10-27 03:00"))

# canonical channel -> regex on the column name (unit suffix stripped of the per-unit "_sensor_N" tag)
CO2_PPM = r"^CO2\s*[\[(]ppm[\])]"
TEMP = r"^(T|Temperature)\s*[\[(]?\s*\W?C\b"
RH = r"^(RH|rH|Humidity)\s*[\[(]?\s*%"
TVOC_PPB = r"^TVOC\s*[\[(]ppb[\])]"
TVOC_PPM = r"^TVOC\s*\(ppm\)"

# provider -> (file, {channel: (regex, scale to canonical unit)}, notes)
PROVIDERS: dict[str, tuple[str, dict[str, tuple[str, float]], list[str]]] = {
    "provider1": ("Indoor_sensor_provider_1.csv",
                  {"eco2": (CO2_PPM, 1.0), "temp_c": (TEMP, 1.0), "rh_pct": (RH, 1.0), "tvoc": (TVOC_PPB, 1.0)},
                  ["TVOC header says ppb but the metadata sheet says ppm; values (median ~100, max ~500) fit ppb"]),
    "provider2": ("Indoor_sensor_provider_2.csv",
                  {"eco2": (CO2_PPM, 1.0), "temp_c": (TEMP, 1.0), "rh_pct": (RH, 1.0)},
                  ["TVOC is a unitless VOC index (column 'TVOC (Index)', 0-500), not ppb: left absent",
                   "~10 min cadence"]),
    "provider3": ("Indoor_sensor_provider_3.csv",
                  {"eco2": (CO2_PPM, 1.0), "temp_c": (TEMP, 1.0), "rh_pct": (RH, 1.0), "tvoc": (TVOC_PPB, 1.0)},
                  ["TVOC stated as ppb but runs ~30x higher than the other providers in the same house "
                   "(median ~5800 vs ~150-250 ppb); scale unverified"]),
    "provider4": ("Indoor_sensor_provider_4.csv",
                  {"eco2": (CO2_PPM, 1.0), "temp_c": (TEMP, 1.0), "rh_pct": (RH, 1.0), "tvoc": (TVOC_PPM, 1000.0)},
                  ["TVOC converted from ppm to ppb (x1000)", "~5 s cadence",
                   "CO2 minimum ~300-320 ppm, below outdoor background: possible offset"]),
    "provider5": ("indoor_sensor_provider_5.csv",
                  {"eco2": (CO2_PPM, 1.0), "temp_c": (TEMP, 1.0), "rh_pct": (RH, 1.0)},
                  ["TVOC in mg/m3 (needs an assumed molar mass for ppb): left absent", "~10 min cadence"]),
    "provider6": ("Indoor_sensor_provider_6.csv",
                  {"eco2": (CO2_PPM, 1.0), "temp_c": (TEMP, 1.0), "rh_pct": (RH, 1.0), "tvoc": (TVOC_PPB, 1.0)},
                  ["CO2 range (~500-1200 ppm) is much narrower than the other providers; placement unknown"]),
    "reference": ("Indoor_referent_device.csv",
                  {"eco2": (CO2_PPM, 1.0), "temp_c": (TEMP, 1.0), "rh_pct": (RH, 1.0)},
                  ["Reference-grade instrument, not a low-cost kit",
                   "VOC in mg/m3 (needs an assumed molar mass for ppb): left absent",
                   "Runs until 2024-11-05, past the low-cost campaign"]),
}

UNIT_SUFFIX = re.compile(r"_sensor_(\d+)$")


def _read(path: Path, provider: str) -> pd.DataFrame:
    df = pd.read_csv(path / PROVIDERS[provider][0], encoding="utf-8")
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def _units(columns) -> list[str]:
    return sorted({m.group(1) for c in columns if (m := UNIT_SUFFIX.search(c))}, key=int)


def devices(path: str | Path) -> list[str]:
    """Device ids: 'providerN' for single-unit files, 'providerN_sK' per unit of a multi-unit file."""
    path = Path(path)
    out = []
    for provider in PROVIDERS:
        header = pd.read_csv(path / PROVIDERS[provider][0], nrows=0, encoding="utf-8").columns
        units = _units(header)
        out += [f"{provider}_s{u}" for u in units] if units else [provider]
    return out


def _select(raw: pd.DataFrame, provider: str, unit: str | None) -> pd.DataFrame:
    cols = {c: UNIT_SUFFIX.sub("", c).strip() for c in raw.columns
            if c != "Timestamp" and (unit is None or c.endswith(f"_sensor_{unit}"))}
    df = pd.DataFrame({"ts": pd.to_datetime(raw["Timestamp"], format="%Y-%m-%d %H:%M:%S")})
    for ch, (pattern, scale) in PROVIDERS[provider][1].items():
        hits = [c for c, name in cols.items() if re.search(pattern, name)]
        if len(hits) != 1:
            raise ValueError(f"EDIAQI {provider} unit {unit}: expected one column for {ch}, found {hits}")
        df[ch] = pd.to_numeric(raw[hits[0]], errors="coerce") * scale
    return df


def load(path: str | Path, device: str, max_gap_minutes: float | None = None,
         _raw: pd.DataFrame | None = None) -> tuple[pd.DataFrame, DatasetMeta, dict]:
    path = Path(path)
    provider, _, unit = device.partition("_s")
    if provider not in PROVIDERS:
        raise ValueError(f"unknown EDIAQI device {device!r}; available: {devices(path)}")
    raw = _read(path, provider) if _raw is None else _raw
    units = _units(raw.columns)
    if bool(unit) != bool(units) or (unit and unit not in units):
        raise ValueError(f"unknown EDIAQI device {device!r}; available: {devices(path)}")

    df = _select(raw, provider, unit or None)
    channels = list(PROVIDERS[provider][1])
    df = df[df[channels].notna().any(axis=1)].reset_index(drop=True)  # rows this unit actually reported

    in_dst = (df["ts"] >= DST_AMBIGUOUS[0]) & (df["ts"] < DST_AMBIGUOUS[1])
    repeated_hour = bool(df.loc[in_dst, "ts"].duplicated().any() or (df.loc[in_dst, "ts"].diff().dt.total_seconds() < 0).any())
    reasons = invalid_reasons(df, channels)
    reasons["dst_ambiguous_hour"] = in_dst
    valid = ~pd.concat(reasons, axis=1).any(axis=1)

    step_s = median_step_s(df["ts"])
    gap = default_max_gap(df["ts"]) if max_gap_minutes is None else max_gap_minutes
    out = finalize(df, valid, gap)
    validate(out)
    report = build_report(len(df), reasons, out, gap, step_s)

    meta = DatasetMeta(
        name=f"EDIAQI indoor {device}",
        source="Zenodo record 14763137 (doi:10.5281/zenodo.14763137), data.zip",
        license="CC BY 4.0",
        device_id=device,
        cadence_note=f"median native step {step_s:.0f} s",
        timezone="Europe/Zagreb local wall clock (metadata: 'Zagreb timezone (CET)'); "
                 "ambiguous DST hour 2024-10-27 02:00-03:00 dropped",
        unit_assumptions={
            "eco2": "CO2 in ppm as reported; NDIR vs MOx eCO2 not stated",
            "temp_c": "degC (stated)",
            "rh_pct": "percent (stated)",
            **({"tvoc": "ppb (stated)" if PROVIDERS[provider][1]["tvoc"][1] == 1.0 else "ppb, converted from ppm"}
               if "tvoc" in channels else {}),
        },
        notes=[*PROVIDERS[provider][2],
               "repeated hour seen at the DST change (clock follows local DST)" if repeated_hour else
               "no repeated hour at the DST change: clock may be fixed-offset or UTC (unverified)",
               "No AQI channel in EDIAQI"],
        verified_against_real_file=True,
    )
    return out, meta, report


def load_all(path: str | Path, max_gap_minutes: float | None = None) -> dict[str, tuple[pd.DataFrame, DatasetMeta, dict]]:
    """Every device, reading each provider file once."""
    path = Path(path)
    out, cache = {}, {}
    for device in devices(path):
        provider = device.partition("_s")[0]
        if provider not in cache:
            cache[provider] = _read(path, provider)
        out[device] = load(path, device, max_gap_minutes, _raw=cache[provider])
    return out
