"""Alert evaluation at a fixed false-alarm budget, with the operating point chosen on training data only.

Same design as the learning curve (scripts/run_learning_curve.py): per device, train on exactly D days, test on the next
day, common test days for every D, purge of horizon + lag window, calendar features. For each split:

  * threshold  = percentile q of the target over the D training days (--alert-q, e.g. 90 and 95);
  * operating point: the training windows are split in time into a head and a validation tail (--val-frac, with the
    same gap the models use). Each model is fit on the head and predicts the tail; the decision offset with the best
    recall at FA/day <= budget on the tail is kept (alerts.choose_offset). The model is then refit on all D days and
    evaluated on the test day with that offset. Nothing about the test day is used to choose anything.
  * mode "raw": offset 0 (alarm when the forecast exceeds the threshold), no tuning, for reference.

Persistence with a tuned offset is the "threshold on the current value" baseline: it alarms when y(t) is within
the offset below the threshold (a positive offset alarms earlier and more often). With --events (DALTON only), each test event is flagged when an activity annotation lies
within [onset - 60 min, onset + 15 min] after --ann-shift-min (see scripts/check_annotation_clock.py), and hits on
annotated events are counted separately.

alarm_frac = share of test origins in the alarm state: consecutive alarms merge into one notification, so a model that
alarms all day counts as one false alarm; alarm_frac exposes that.

One CSV row per (device, D, split, model, q, mode, budget). Summaries: scripts/summarize_alerts.py.

  python scripts/run_alerts.py --dataset dalton --path data/raw/dalton --devices H1_41 R1_51 --target eco2 \
      --horizon 30 --days 3 7 14 --workers 4 --out results/alerts/dalton_eco2_h30.csv
"""
from __future__ import annotations

import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from coldstart import alerts as al
from coldstart import evaluate as ev
from coldstart.cache import DEFAULT_DIR, load_grid
from coldstart.models import REGISTRY, PolicyDelta

LEARNED = ("ridge", "huber", "lightgbm", "random_forest")
MIN_HEAD, MIN_VAL = 50, 12  # windows; shorter histories get no tuned rows


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--path", required=True)
    p.add_argument("--devices", nargs="+", required=True, help="device ids, or 'all' for every device of the adapter")
    p.add_argument("--target", default="eco2")
    p.add_argument("--cadence", type=int, default=5)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--lags", type=int, default=6)
    p.add_argument("--days", type=float, nargs="+", default=[3, 7, 14])
    p.add_argument("--test-days", type=float, default=1.0)
    p.add_argument("--models", nargs="+", default=["persistence", "ridge", "huber", "lightgbm", "policy"])
    p.add_argument("--alert-q", type=float, nargs="+", default=[90, 95])
    p.add_argument("--budgets", type=float, nargs="+", default=[0.5, 1, 2], help="false alarms per day")
    p.add_argument("--val-frac", type=float, default=0.3, help="training tail used to choose the offset")
    p.add_argument("--events", action="store_true", help="flag test events that match an activity annotation (DALTON)")
    p.add_argument("--ann-shift-min", type=float, default=0, help="minutes added to annotation times before matching")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def head_val(tr: np.ndarray, t0: np.ndarray, val_frac: float, gap: int):
    """Time-ordered head and validation tail of the training windows, `gap` windows apart."""
    tr = tr[np.argsort(t0[tr], kind="stable")]
    n_val = int(val_frac * len(tr))
    head, val = tr[: max(0, len(tr) - n_val - gap)], tr[len(tr) - n_val:]
    return head, val


def _fit_predict(names, X, w, fit_idx, pred_idx, gap):
    """Predictions on pred_idx for each model fit on fit_idx (persistence needs no fit; policy reuses ridge/huber)."""
    need = set(names) | (set(PolicyDelta.CANDIDATES) if "policy" in names else set())
    fitted = {m: REGISTRY[m](seed=0, gap=gap).fit(X[fit_idx], w.y[fit_idx], w.y0[fit_idx])
              for m in LEARNED if m in need}
    out, choice = {}, ""
    for m in names:
        if m == "persistence":
            out[m] = w.y0[pred_idx].astype(float)
        elif m == "policy":
            pol = PolicyDelta.from_fitted({c: fitted[c] for c in PolicyDelta.CANDIDATES})
            out[m], choice = pol.predict(X[pred_idx], w.y0[pred_idx]), pol.choice_
        else:
            out[m] = fitted[m].predict(X[pred_idx], w.y0[pred_idx])
    return out, choice


def _event_cols(r: dict, ann_ts, shift) -> dict:
    if ann_ts is None:
        return {}
    flag = al.annotated(r["event_onsets"], ann_ts, shift_min=shift)
    return {"n_events_annotated": int(flag.sum()), "hits_annotated": int((flag & r["event_hit"]).sum())}


def run_device(a, device: str) -> list[dict]:
    grid, chs, _ = load_grid(a.dataset, a.path, device, a.target, a.cadence, DEFAULT_DIR)
    hs = a.horizon // a.cadence
    w = ev.make_windows(grid, a.target, chs, a.lags, hs)
    if len(w.y) == 0:
        return []
    X = np.hstack([w.X, ev.calendar_features(w.t0)])
    gap, purge = a.lags + hs, a.horizon + a.lags * a.cadence
    span = (w.t_target.max() - w.t0.min()) / np.timedelta64(1, "D")
    days = [D for D in a.days if D + a.test_days <= span]
    ann_ts = None
    if a.events:
        from coldstart.adapters import dalton
        ann_ts = dalton.load_events(a.path, device.split("_")[0])["ts"].to_numpy()
    rows = []
    for D in days:
        for split, (tr, te) in enumerate(ev.rolling_origin(w, D, a.test_days, purge, step_days=1.0,
                                                               first_origin_days=max(days))):
            t_end = w.t_target[tr].max()
            tr = tr[w.t0[tr] >= t_end - np.timedelta64(int(D * 86400), "s")]
            head, val = head_val(tr, w.t0, a.val_frac, gap)
            tuned = len(head) >= MIN_HEAD and len(val) >= MIN_VAL
            test_pred, choice = _fit_predict(a.models, X, w, tr, te, gap)
            val_pred, val_choice = _fit_predict(a.models, X, w, head, val, gap) if tuned else ({}, "")
            offsets = al.offset_grid(w.y[tr] - w.y0[tr])
            span_te = (w.t_target[te].max() - w.t_target[te].min()) / np.timedelta64(1, "D")
            base = {"dataset": a.dataset, "device": device, "target": a.target, "cadence_min": a.cadence,
                    "horizon_min": a.horizon, "train_days": D, "split": split, "n_train": len(tr), "n_head": len(head),
                    "n_val": len(val), "n_test": len(te), "test_start": str(w.t0[te].min()), "span_days": span_te}
            for m in a.models:
                for q in a.alert_q:
                    thr = al.percentile_threshold(w.y[tr], q)
                    mb = {**base, "model": m, "alert_q": q, "threshold": thr,
                          "policy_choice": choice if m == "policy" else ""}
                    r = al.evaluate_alerts(w.t0[te], w.t_target[te], w.y[te], test_pred[m], thr, a.horizon,
                                           return_events=True)
                    rows.append({**mb, "mode": "raw", "budget": np.nan, "offset": 0.0, **_core(r),
                                 "alarm_frac": float(np.mean(test_pred[m] > thr)),
                                 **_event_cols(r, ann_ts, a.ann_shift_min)})
                    if not tuned:
                        continue
                    for b in a.budgets:
                        off, rv = al.choose_offset(w.t0[val], w.t_target[val], w.y[val], val_pred[m], thr, a.horizon,
                                                   b, offsets)
                        r = al.evaluate_alerts(w.t0[te], w.t_target[te], w.y[te], test_pred[m], thr, a.horizon,
                                               offset=off, return_events=True)
                        rows.append({**mb, "mode": "tuned", "budget": b, "offset": off, **_core(r),
                                     "alarm_frac": float(np.mean(test_pred[m] + off > thr)),
                                     "val_recall": rv["recall"], "val_fa_per_day": rv["fa_per_day"],
                                     "val_policy_choice": val_choice if m == "policy" else "",
                                     **_event_cols(r, ann_ts, a.ann_shift_min)})
        print(f"{device} D={D} done", flush=True)
    return rows


def _core(r: dict) -> dict:
    return {k: r[k] for k in ("n_events", "hits", "recall", "false_alarms", "fa_per_day", "median_lead_min")}


def _safe(args):
    a, device = args
    try:
        return run_device(a, device)
    except Exception:  # noqa: BLE001 - one device must not stop the others
        traceback.print_exc()
        print(f"{device} FAILED", flush=True)
        return []


def main(argv=None):
    a = parse(argv)
    if a.devices == ["all"]:
        import importlib
        a.devices = importlib.import_module(f"coldstart.adapters.{a.dataset}").devices(a.path)
    jobs = [(a, d) for d in a.devices]
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as pool:
            parts = list(pool.map(_safe, jobs))
    else:
        parts = [_safe(j) for j in jobs]
    rows = [r for p in parts for r in p]
    if not rows:
        raise SystemExit("no rows")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
