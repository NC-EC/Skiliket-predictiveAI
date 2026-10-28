"""Canonical schema, aligned with OGC SensorThings (ObservedProperty / Datastream / Observation).

Every adapter returns a tidy DataFrame with CANONICAL_COLUMNS plus a metadata dict.
Rules (see proposal): units converted to canonical, timestamps kept explicit about their
time zone status, invalid readings flagged (never imputed), gaps split series into segments.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# channel -> (unit, SensorThings ObservedProperty name, description)
CHANNELS: dict[str, tuple[str, str, str]] = {
    "eco2": ("ppm", "equivalent CO2", "Equivalent CO2 estimated by the sensor (not a direct NDIR reading unless stated)"),
    "tvoc": ("ppb", "total volatile organic compounds", "TVOC concentration estimated by the sensor"),
    "aqi": ("index_1_5", "air quality index (sensor-defined, 1-5)", "Sensor-defined index, 1 = excellent .. 5 = unhealthy"),
    "temp_c": ("degC", "air temperature", "Ambient air temperature"),
    "rh_pct": ("percent", "relative humidity", "Relative humidity"),
}

CANONICAL_COLUMNS = ["ts", "segment", "valid", *CHANNELS]


@dataclass
class DatasetMeta:
    """FAIR-style metadata carried with every loaded dataset."""

    name: str
    source: str
    license: str
    device_id: str
    cadence_note: str
    timezone: str = "unknown"          # keep 'unknown' until confirmed; never silently assume
    unit_assumptions: dict[str, str] = field(default_factory=dict)  # channel -> note (e.g. 'inferred')
    notes: list[str] = field(default_factory=list)
    verified_against_real_file: bool = False


def validate(df: pd.DataFrame) -> None:
    """Raise ValueError if df does not follow the canonical layout."""
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing canonical columns: {missing}")
    if not df["ts"].is_monotonic_increasing:
        raise ValueError("ts must be sorted ascending")
    if df["ts"].duplicated().any():
        raise ValueError("duplicated timestamps")
    if not df["segment"].is_monotonic_increasing:
        raise ValueError("segment ids must be non-decreasing")
    bad = df[df["valid"]]
    for c in CHANNELS:
        if bad[c].isna().any():
            raise ValueError(f"valid rows contain NaN in {c}")
