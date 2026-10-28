"""Alert evaluation: does a forecast warn before the target crosses a threshold?

Definitions (per proposal)
  * Threshold: per-device percentile (p90/p95) of the target computed on TRAINING data only.
  * Event: a maximal run of grid points where the observed target > threshold, separated from the
    next run by at least `min_sep_min`. Event onset = first point of the run.
  * Alarm: forecast issued at origin t predicts y(t+h) > threshold. Consecutive alarm origins
    closer than `min_sep_min` are merged (one notification).
  * Hit: an event whose onset lies in (alarm_time, alarm_time + h + tolerance]; lead time = onset - alarm_time.
  * False alarm: a merged alarm that hits no event and is not issued while already inside an event.
  * Report recall, false alarms per day, median lead time. To compare models at a fixed false-alarm
    budget, sweep a decision offset on the forecast and pick the best recall with FA/day <= budget.
"""
from __future__ import annotations

import numpy as np


def percentile_threshold(train_values: np.ndarray, q: float) -> float:
    return float(np.percentile(train_values, q))


def _runs(times: np.ndarray, mask: np.ndarray, min_sep: np.timedelta64) -> list[tuple]:
    t = times[mask]
    if len(t) == 0:
        return []
    out, start, prev = [], t[0], t[0]
    for x in t[1:]:
        if x - prev >= min_sep:
            out.append((start, prev))
            start = x
        prev = x
    out.append((start, prev))
    return out


def evaluate_alerts(t0, t_target, y_true, y_pred, threshold, horizon_min, min_sep_min=15, tolerance_min=0, offset=0.0):
    """t0/t_target: origin and target times of test windows; y_true at target; y_pred forecast for target."""
    order = np.argsort(t_target)
    tt, yt = t_target[order], y_true[order]
    events = _runs(tt, yt > threshold, np.timedelta64(min_sep_min, "m"))
    o2 = np.argsort(t0)
    alarms = _runs(t0[o2], (y_pred[o2] + offset) > threshold, np.timedelta64(min_sep_min, "m"))
    h = np.timedelta64(int(horizon_min + tolerance_min), "m")

    hit_events, leads, false_alarms = set(), [], 0
    for a_start, _ in alarms:
        inside = any(s <= a_start <= e for s, e in events)
        matched = [i for i, (s, _) in enumerate(events) if a_start < s <= a_start + h]
        if matched:
            for i in matched:
                if i not in hit_events:
                    hit_events.add(i)
                    leads.append((events[i][0] - a_start) / np.timedelta64(1, "m"))
        elif not inside:
            false_alarms += 1

    span_days = max((tt.max() - tt.min()) / np.timedelta64(1, "D"), 1e-9) if len(tt) else np.nan
    return {
        "n_events": len(events),
        "hits": len(hit_events),
        "recall": len(hit_events) / len(events) if events else np.nan,
        "false_alarms": false_alarms,
        "fa_per_day": false_alarms / span_days,
        "median_lead_min": float(np.median(leads)) if leads else np.nan,
    }


def recall_at_budget(t0, t_target, y_true, y_pred, threshold, horizon_min, budget_per_day=1.0, offsets=None, **kw):
    """Best recall over decision offsets subject to FA/day <= budget. Returns metrics dict with chosen offset.

    NOTE: choosing the offset on the test period is optimistic; in the experiments choose it on a
    validation tail of the training data and pass it via `offset` to evaluate_alerts.
    """
    if offsets is None:
        scale = np.std(y_true) if len(y_true) else 1.0
        offsets = np.linspace(-scale, scale, 41)
    best = None
    for off in offsets:
        r = evaluate_alerts(t0, t_target, y_true, y_pred, threshold, horizon_min, offset=off, **kw)
        if r["fa_per_day"] <= budget_per_day and not np.isnan(r["recall"]):
            if best is None or r["recall"] > best["recall"]:
                best = {**r, "offset": float(off)}
    return best
