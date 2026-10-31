"""Pretrained zero-shot forecasters (Chronos family) on the learning-curve test days: the D = 0 point.

The model is never trained on the device. At each test origin it gets the last --context-hours of the target (inside
the origin's segment, nothing filled in) and forecasts up to the largest horizon; each horizon reads its step. The test
days are the learning curve's common test days (those of the largest feasible D in --days), so the rows are directly
comparable with the runner's rows for the same device and horizon. Persistence rows are written too, so the CSV can go
through scripts/summarize.py (train_days = 0).

Cost columns: n_params of the pretrained model, infer_ms = median wall time per window over the batched calls, and the
process peak RSS after loading the model.

  python scripts/run_zeroshot.py --dataset dalton --path data/raw/dalton --devices all --target eco2 \
      --model amazon/chronos-bolt-tiny --alias bolt_tiny --context-hours 2 24 --out results/zeroshot/dalton_eco2_bolt_tiny.csv

Each device is saved to <out>_parts/<device>.csv when it finishes; re-running skips finished devices.
"""
from __future__ import annotations

import argparse
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from coldstart import evaluate as ev
from coldstart import zeroshot as zs
from coldstart.cache import DEFAULT_DIR, load_grid
from coldstart.cost import _max_rss_mb


def load_pipeline(model_id: str):
    """Chronos pipeline on CPU in float32 (chronos-forecasting >= 2.0). Weights come from the Hugging Face hub."""
    import torch
    from chronos import BaseChronosPipeline

    return BaseChronosPipeline.from_pretrained(model_id, device_map="cpu", dtype=torch.float32)


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--path", required=True)
    p.add_argument("--devices", nargs="+", required=True, help="device ids, or 'all'")
    p.add_argument("--target", default="eco2")
    p.add_argument("--cadence", type=int, default=5)
    p.add_argument("--horizons", type=int, nargs="+", default=[15, 30, 60], help="minutes")
    p.add_argument("--lags", type=int, default=6, help="only to build the same windows and purge as the runner")
    p.add_argument("--days", type=float, nargs="+", default=[1, 2, 3, 5, 7, 14],
                   help="the learning curve's train lengths; the test days are those of the largest feasible one")
    p.add_argument("--test-days", type=float, default=1.0)
    p.add_argument("--model", default="amazon/chronos-bolt-tiny")
    p.add_argument("--alias", default="bolt_tiny", help="model name in the CSV: chronos_<alias>_ctx<h>h")
    p.add_argument("--context-hours", type=float, nargs="+", default=[2, 24])
    p.add_argument("--covariates", action="store_true", help="pass the other channels as past covariates (Chronos-2)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def test_splits(w, days, test_days, purge):
    """(split index, test window indices) for the common test days of the largest feasible D."""
    span = (w.t_target.max() - w.t0.min()) / np.timedelta64(1, "D") if len(w.y) else 0.0
    feasible = [D for D in days if D + test_days <= span]
    if not feasible:
        return []
    D = max(feasible)
    return [(k, te) for k, (_, te) in enumerate(ev.rolling_origin(w, D, test_days, purge, step_days=1.0,
                                                                   first_origin_days=D))]


def run_device(a, pipe, n_params, device):
    grid, chs, _ = load_grid(a.dataset, a.path, device, a.target, a.cadence, DEFAULT_DIR)
    cov = [c for c in chs if c != a.target] if a.covariates else None
    per_h = {}
    for h in a.horizons:
        if h % a.cadence:
            continue
        w = ev.make_windows(grid, a.target, chs, a.lags, h // a.cadence)
        if len(w.y):
            per_h[h] = (w, test_splits(w, a.days, a.test_days, h + a.lags * a.cadence))
    origins = sorted({t for w, sp in per_h.values() for _, te in sp for t in w.t0[te]})
    if not origins:
        return []
    origins = np.array(origins, dtype=grid["ts"].to_numpy().dtype)
    steps = max(per_h) // a.cadence
    preds = {}
    for ch in a.context_hours:
        n_ctx = max(1, int(round(ch * 60 / a.cadence)))
        ctx = zs.contexts(grid, a.target, origins, n_ctx, cov)
        t = time.perf_counter()
        f = zs.median_forecast(pipe, ctx, steps, a.batch_size)
        ms = (time.perf_counter() - t) * 1000 / max(len(ctx), 1)
        preds[ch] = (pd.DataFrame(f, index=pd.DatetimeIndex(origins)), ms)
    rows = []
    for h, (w, splits) in per_h.items():
        k = h // a.cadence
        for split, te in splits:
            y, y0 = w.y[te], w.y0[te]
            pmae = ev.mae(y, y0)
            base = {"dataset": a.dataset, "device": device, "target": a.target, "cadence_min": a.cadence,
                    "horizon_min": h, "train_days": 0, "split": split, "n_train": 0, "n_test": len(te),
                    "test_start": str(w.t0[te].min()), "seed": 0, "host": platform.machine(), "parallel": 1}
            rows.append({**base, "model": "persistence", "mae": pmae, "rmse": ev.rmse(y, y0), "rel_mae": 1.0})
            for ch, (P, ms) in preds.items():
                yp = P.loc[pd.DatetimeIndex(w.t0[te])].to_numpy()[:, k - 1]
                rows.append({**base, "model": f"chronos_{a.alias}_ctx{ch:g}h", "context_steps": int(round(ch * 60 / a.cadence)),
                             "mae": ev.mae(y, yp), "rmse": ev.rmse(y, yp),
                             "rel_mae": ev.mae(y, yp) / max(pmae, 1e-12), "infer_ms": ms, "n_params": n_params,
                             "max_rss_mb": _max_rss_mb()})
    return rows


def main(argv=None):
    a = parse(argv)
    if a.devices == ["all"]:
        import importlib
        a.devices = importlib.import_module(f"coldstart.adapters.{a.dataset}").devices(a.path)
    parts = Path(a.out).with_name(Path(a.out).stem + "_parts")
    parts.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    pipe = load_pipeline(a.model)
    n_params = zs.n_params(pipe)
    print(f"{a.model}: {n_params} parameters, loaded in {time.perf_counter() - t:.1f} s, RSS {_max_rss_mb():.0f} MB",
          flush=True)
    for d in a.devices:
        part = parts / f"{d}.csv"
        if part.exists():
            print(f"{d}: already done", flush=True)
            continue
        try:
            rows = run_device(a, pipe, n_params, d)
        except Exception:  # noqa: BLE001 - one device must not stop the others
            traceback.print_exc()
            print(f"{d} FAILED", flush=True)
            continue
        if rows:
            tmp = part.with_suffix(".tmp")
            pd.DataFrame(rows).to_csv(tmp, index=False)
            tmp.replace(part)
        print(f"{d}: {len(rows)} rows", flush=True)
    found = [parts / f"{d}.csv" for d in a.devices if (parts / f"{d}.csv").exists()]
    if not found:
        raise SystemExit("no rows")
    out = pd.concat([pd.read_csv(f) for f in found], ignore_index=True)
    out.to_csv(a.out, index=False)
    print(f"{len(out)} rows from {len(found)} of {len(a.devices)} devices -> {a.out}")


if __name__ == "__main__":
    main()
