"""Diagnose learned models scoring worse than persistence (e.g. EDIAQI provider1 MASE ~2-3 vs 1.2).

For every split it prints what separates a code bug from a genuine result:
  align   : y0 must equal the last target lag in X (window alignment). Must be 0.
  denom   : in-sample persistence MAE (MASE denominator). Small values inflate the mean of MASE across splits.
  in_rel  : model MAE / persistence MAE on the TRAINING windows. If > 1 the model cannot even fit the training
            data, which points to a bug (scaling, target, alignment).
  out_rel : the same ratio on the TEST windows (relative MAE, a scale-free skill score per split).
  shift   : share of test rows with some feature outside the training range (extrapolation risk).
  d_tr/d_te: std of the target change (y - y0) in train vs test.

Usage:
  python scripts/diagnose_mase.py --dataset ediaqi --path data/raw/ediaqi --device provider1 --target eco2 --days 1 2 3
"""
from __future__ import annotations

import argparse
import importlib

import numpy as np

from coldstart import evaluate as ev
from coldstart.models import REGISTRY

CHANNELS = ["eco2", "tvoc", "temp_c", "rh_pct"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--path", required=True)
    p.add_argument("--device")
    p.add_argument("--target", default="eco2")
    p.add_argument("--cadence", type=int, default=5)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--lags", type=int, default=6)
    p.add_argument("--days", type=float, nargs="+", default=[1, 2, 3])
    p.add_argument("--models", nargs="+", default=["ridge", "random_forest", "lightgbm"])
    a = p.parse_args()

    adapter = importlib.import_module(f"coldstart.adapters.{a.dataset}")
    df, meta, rep = adapter.load(a.path, a.device) if a.device else adapter.load(a.path)
    chs = [a.target] + [c for c in CHANNELS if c != a.target and df[c].notna().all()]
    grid = ev.to_grid(df, a.cadence, chs)
    w = ev.make_windows(grid, a.target, chs, a.lags, a.horizon // a.cadence)
    C = len(chs)
    align = np.abs(w.y0 - w.X[:, (a.lags - 1) * C]).max()
    print(f"{meta.name}: channels={chs} grid={len(grid)} windows={len(w.y)} segments={grid.segment.nunique()}")
    print(f"align max|y0 - last target lag| = {align:.3g}  (must be 0)")
    print(f"report: {rep}")
    purge = a.horizon + a.lags * a.cadence

    for D in a.days:
        for split, (tr, te) in enumerate(ev.rolling_origin(w, D, 1.0, purge, step_days=1.0)):
            t_end = w.t_target[tr].max()
            tr = tr[w.t0[tr] >= t_end - np.timedelta64(int(D * 86400), "s")]
            p_tr = np.mean(np.abs(w.y[tr] - w.y0[tr]))
            p_te = np.mean(np.abs(w.y[te] - w.y0[te]))
            lo, hi = w.X[tr].min(0), w.X[tr].max(0)
            shift = np.mean(((w.X[te] < lo) | (w.X[te] > hi)).any(1))
            line = (f"D={D:g} s={split} ntr={len(tr)} nte={len(te)} denom={p_tr:7.2f} pers_te={p_te:7.2f} "
                    f"d_tr={np.std(w.y[tr]-w.y0[tr]):6.1f} d_te={np.std(w.y[te]-w.y0[te]):6.1f} shift={shift:.2f}")
            for m in a.models:
                mdl = REGISTRY[m](seed=0).fit(w.X[tr], w.y[tr], w.y0[tr])
                in_rel = np.mean(np.abs(w.y[tr] - mdl.predict(w.X[tr], w.y0[tr]))) / p_tr
                out_rel = np.mean(np.abs(w.y[te] - mdl.predict(w.X[te], w.y0[te]))) / p_te
                line += f" | {m[:5]} in={in_rel:.2f} out={out_rel:.2f}"
            print(line)


if __name__ == "__main__":
    main()
