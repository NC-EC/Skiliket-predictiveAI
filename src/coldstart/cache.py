"""On-disk cache of the regular grid per (dataset, device, target, cadence), so 1 Hz CSVs are read once.

The key includes a hash of the code that produces the grid (the adapter, adapters/base.py, evaluate.py), so editing
the cleaning rules or the gridding invalidates old entries. Files live in data/cache/ (gitignored).
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import pickle
from pathlib import Path

import pandas as pd

from coldstart import evaluate as ev
from coldstart.adapters import base

CHANNELS = ["eco2", "tvoc", "temp_c", "rh_pct"]
DEFAULT_DIR = Path(__file__).parents[2] / "data" / "cache"


def _code_hash(adapter) -> str:
    src = "".join(inspect.getsource(m) for m in (adapter, base, ev))
    return hashlib.sha1(src.encode()).hexdigest()[:10]


def build_grid(df: pd.DataFrame, target: str, cadence: int) -> tuple[pd.DataFrame, list[str]]:
    """Target first, then every covariate present on all rows; grid by mean per segment (evaluate.to_grid)."""
    chs = [target] + [c for c in CHANNELS if c != target and df[c].notna().all()]
    return ev.to_grid(df, cadence, chs), chs


def load_grid(dataset: str, path: str, device: str | None, target: str, cadence: int,
              cache_dir: str | Path | None = DEFAULT_DIR) -> tuple[pd.DataFrame, list[str], str]:
    """Return (grid, channels, dataset name). cache_dir=None disables the cache."""
    adapter = importlib.import_module(f"coldstart.adapters.{dataset}")
    f = None
    if cache_dir is not None:
        f = Path(cache_dir) / f"{dataset}_{device or 'all'}_{target}_{cadence}min_{_code_hash(adapter)}.pkl"
        if f.exists():
            with open(f, "rb") as fh:
                c = pickle.load(fh)
            if c["path"] == str(Path(path).resolve()):
                return c["grid"], c["channels"], c["name"]
    df, meta, _ = adapter.load(path, device) if device else adapter.load(path)
    if df[target].isna().all():
        raise SystemExit(f"{meta.name} has no '{target}' channel")
    grid, chs = build_grid(df, target, cadence)
    if f is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "wb") as fh:
            pickle.dump({"path": str(Path(path).resolve()), "grid": grid, "channels": chs, "name": meta.name}, fh)
    return grid, chs, meta.name
