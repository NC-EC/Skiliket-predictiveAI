"""Cold-start learning curve: train each model on exactly D days of history, test on the next day.

Each origin simulates a kit that has been running for D days. Origins slide forward by one day, so
every (D, origin) pair is a replicate. Writes one CSV row per (model, seed, D, split).

Example:
  python scripts/run_learning_curve.py --dataset skiliket --path data/raw/skiliket_zenodo_17398551.xlsx \
      --target eco2 --cadence 5 --horizon 15 --days 1 2 3 --seeds 3 --out results/smoke_skiliket.csv
  python scripts/run_learning_curve.py --dataset ediaqi --path data/raw/ediaqi --device provider1 \
      --target eco2 --cadence 5 --horizon 15 --days 1 2 3 --seeds 3 --out results/smoke_ediaqi_provider1.csv
"""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from coldstart import evaluate as ev
from coldstart.cache import DEFAULT_DIR, load_grid
from coldstart.alerts import evaluate_alerts, percentile_threshold
from coldstart.cost import measure_fit, measure_inference
from coldstart.cost import Cost
from coldstart.models import REGISTRY, PolicyDelta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="adapter module name in coldstart.adapters")
    p.add_argument("--path", required=True)
    p.add_argument("--device", help="device id for multi-device datasets (e.g. ediaqi provider1, dalton H1_41, urbino olivetti)")
    p.add_argument("--target", default="eco2")
    p.add_argument("--cadence", type=int, default=5)
    p.add_argument("--horizon", type=int, default=15, help="minutes")
    p.add_argument("--lags", type=int, default=6, help="number of lag steps")
    p.add_argument("--days", type=float, nargs="+", default=[1, 2, 3])
    p.add_argument("--test-days", type=float, default=1.0)
    p.add_argument("--models", nargs="+", default=["ridge", "huber", "random_forest", "lightgbm", "gru"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--alert-q", type=float, default=90)
    p.add_argument("--common-test", action=argparse.BooleanOptionalAction, default=True,
                   help="test every train length on the same days (those available for the largest --days); "
                        "--no-common-test for quick smoke checks")
    p.add_argument("--calendar", action=argparse.BooleanOptionalAction, default=True,
                   help="add hour-of-day sin/cos and a weekend flag (from the origin time) to the learned models' inputs")
    p.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True,
                   help="reuse the gridded series from data/cache/ instead of re-reading the raw files")
    p.add_argument("--parallel", type=int, default=1,
                   help="number of runner processes sharing the machine; > 1 means the cost columns are not timing")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    grid, chs, name = load_grid(a.dataset, a.path, a.device, a.target, a.cadence, DEFAULT_DIR if a.cache else None)
    if a.horizon % a.cadence:
        raise SystemExit("horizon must be a multiple of cadence")
    w = ev.make_windows(grid, a.target, chs, a.lags, a.horizon // a.cadence)
    purge = a.horizon + a.lags * a.cadence
    # windows closer than this many steps share grid points; validation inside a model skips them
    gap = a.lags + a.horizon // a.cadence
    # learned models get calendar columns appended; baselines keep reading the lag block of w.X
    X = np.hstack([w.X, ev.calendar_features(w.t0)]) if a.calendar else w.X
    n_static = len(ev.CALENDAR) if a.calendar else 0
    print(f"{name}: grid={len(grid)} windows={len(w.y)} channels={chs} calendar={a.calendar}")
    # train lengths this device can hold (D days of training plus one test period); with --common-test the largest
    # of these sets the first origin, so a short device keeps its feasible lengths instead of losing every split
    span_days = (w.t_target.max() - w.t0.min()) / np.timedelta64(1, "D") if len(w.y) else 0.0
    days = [D for D in a.days if D + a.test_days <= span_days]
    if days != list(a.days):
        print(f"span {span_days:.1f} days: train lengths {sorted(set(a.days) - set(days))} skipped")
    if not days:
        raise SystemExit(f"{name}: {span_days:.1f} days of windows, too short for any --days")

    rows = []
    for D in days:
        for split, (tr, te) in enumerate(ev.rolling_origin(w, D, a.test_days, purge, step_days=1.0,
                                                               first_origin_days=max(days) if a.common_test else None)):
            # fixed-length history: keep only the last D days before the train end
            t_end = w.t_target[tr].max()
            tr = tr[w.t0[tr] >= t_end - np.timedelta64(int(D * 86400), "s")]
            thr = percentile_threshold(w.y[tr], a.alert_q)
            base = {"dataset": a.dataset, "device": a.device or "", "target": a.target, "cadence_min": a.cadence, "horizon_min": a.horizon,
                    "calendar": a.calendar, "common_test": a.common_test, "train_days": D, "split": split, "n_train": len(tr), "n_test": len(te),
                    "test_start": str(w.t0[te].min()), "host": platform.machine(), "parallel": a.parallel}
            # baselines (seed-independent)
            preds = {"persistence": ev.persistence(w, te), "moving_average": ev.moving_average(w, te, len(chs))}
            for name, yp in preds.items():
                rows.append({**base, "model": name, "seed": 0, **_metrics(w, tr, te, yp, thr, a)})
            fitted = {}  # (model, seed) -> (model, cost); the policy reuses seed 0 of its candidates
            # the policy goes last so its candidates are already fit
            for mname in sorted(a.models, key=lambda m: m == "policy"):
                for seed in range(a.seeds if mname in ("random_forest", "lightgbm", "gru") else 1):
                    kw = {"gap": gap, **({"n_channels": len(chs), "n_static": n_static} if mname == "gru" else {})}
                    if mname == "policy":
                        for c in PolicyDelta.CANDIDATES:  # candidates not in --models are fit (without rows)
                            if (c, 0) not in fitted:
                                fitted[c, 0] = measure_fit(lambda: REGISTRY[c](seed=0, gap=gap).fit(X[tr], w.y[tr], w.y0[tr]))
                        model, cost = _policy(fitted)
                    else:
                        model, cost = measure_fit(lambda: REGISTRY[mname](seed=seed, **kw).fit(X[tr], w.y[tr], w.y0[tr]))
                        fitted[mname, seed] = model, cost
                    yp = model.predict(X[te], w.y0[te])
                    infer = _infer_ms(model, X[te], w.y0[te])
                    rows.append({**base, "model": mname, "seed": seed, **_metrics(w, tr, te, yp, thr, a),
                                 **cost.asdict(), "infer_ms": infer, "policy_choice": getattr(model, "choice_", "")})
            print(f"D={D} split={split} done")
    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(out.groupby(["train_days", "model"])["mase"].mean().unstack().round(3).to_string())


def _policy(fitted):
    """Policy from the candidates already fit: its training cost is theirs (time summed, memory the largest peak)."""
    cands = {c: fitted[c, 0] for c in PolicyDelta.CANDIDATES}
    model = PolicyDelta.from_fitted({c: m for c, (m, _) in cands.items()})
    costs = [c for _, c in cands.values()]
    chosen = cands.get(model.choice_)
    cost = Cost(sum(c.wall_s for c in costs), sum(c.cpu_s for c in costs), max(c.peak_py_mb for c in costs),
                max(c.max_rss_mb for c in costs), chosen[1].model_bytes if chosen else 0)
    return model, cost


def _metrics(w, tr, te, yp, thr, a):
    m = {"mase": ev.mase(w.y[te], yp, w.y[tr], w.y0[tr]), "mae": ev.mae(w.y[te], yp), "rmse": ev.rmse(w.y[te], yp),
         # scale-free skill on the same test split: MAE(model) / MAE(persistence); < 1 beats persistence.
         # Unlike the mean of MASE across splits, it is not inflated by splits with a tiny training denominator.
         "rel_mae": ev.mae(w.y[te], yp) / max(ev.mae(w.y[te], w.y0[te]), 1e-12)}
    al = evaluate_alerts(w.t0[te], w.t_target[te], w.y[te], yp, thr, a.horizon)
    return {**m, "alert_threshold": thr, **{f"alert_{k}": v for k, v in al.items()}}


def _infer_ms(model, X, y0, repeats=30):
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        model.predict(X[:1], y0[:1])
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


if __name__ == "__main__":
    main()
