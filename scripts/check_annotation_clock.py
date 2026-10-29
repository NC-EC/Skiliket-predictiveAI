"""Do DALTON activity annotations and the sensor clock agree? Scan clock shifts before using annotations.

For each device: threshold events of the target (percentile --q over the device's whole series; descriptive only, no
forecasting) and the share of event onsets with an annotation of the device's site within [onset - 60 min,
onset + 15 min], for annotation shifts from -12 h to +12 h. If both clocks agree, the share peaks near 0; a peak at
e.g. +330 min would mean the annotations are in IST and the sensor clock in UTC. The share far from the peak is the
level expected by chance. Writes one row per (device, shift) and prints the peak per device and over all devices.

  python scripts/check_annotation_clock.py --path data/raw/dalton --out results/checks/annotation_clock.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from coldstart import alerts as al
from coldstart.adapters import dalton
from coldstart.cache import DEFAULT_DIR, load_grid


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="data/raw/dalton")
    p.add_argument("--devices", nargs="*", help="default: every device whose site has annotations")
    p.add_argument("--target", default="eco2")
    p.add_argument("--q", type=float, default=90)
    p.add_argument("--step-min", type=int, default=15)
    p.add_argument("--max-shift-min", type=int, default=720)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    shifts = np.arange(-a.max_shift_min, a.max_shift_min + 1, a.step_min)
    devs = a.devices or dalton.devices(a.path)
    rows = []
    for d in devs:
        ann = dalton.load_events(a.path, d.split("_")[0])
        if ann.empty:
            continue
        try:
            grid, _, _ = load_grid("dalton", a.path, d, a.target, 5, DEFAULT_DIR)
        except SystemExit as e:
            print(f"{d}: {e}")
            continue
        if grid.empty:
            continue
        t, y = grid["ts"].to_numpy(), grid[a.target].to_numpy(float)
        on = al.event_onsets(t, y, np.percentile(y, a.q))
        # only annotations inside the device's recording span can match
        a_ts = ann["ts"].to_numpy()
        a_ts = a_ts[(a_ts >= t.min()) & (a_ts <= t.max())]
        share = al.shift_scan(on, a_ts, shifts)
        rows += [{"device": d, "shift_min": s, "share": v, "n_events": len(on), "n_annotations": len(a_ts)}
                 for s, v in zip(shifts, share)]
        if len(on) and len(a_ts):
            print(f"{d}: {len(on)} events, {len(a_ts)} annotations, share at 0 = {share[shifts == 0][0]:.2f}, "
                  f"peak {np.nanmax(share):.2f} at {shifts[np.nanargmax(share)]} min", flush=True)
    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    tot = out[out["n_annotations"] > 0].groupby("shift_min")["share"].median()
    print(f"median over devices: share at 0 = {tot.get(0, np.nan):.2f}, peak {tot.max():.2f} at {tot.idxmax()} min")


if __name__ == "__main__":
    main()
