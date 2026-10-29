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


def evaluate_alerts(t0, t_target, y_true, y_pred, threshold, horizon_min, min_sep_min=15, tolerance_min=0, offset=0.0,
                    return_events=False):
    """t0/t_target: origin and target times of test windows; y_true at target; y_pred forecast for target.
    return_events=True adds event_onsets (array) and event_hit (bool array) to the result."""
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
    out = {
        "n_events": len(events),
        "hits": len(hit_events),
        "recall": len(hit_events) / len(events) if events else np.nan,
        "false_alarms": false_alarms,
        "fa_per_day": false_alarms / span_days,
        "median_lead_min": float(np.median(leads)) if leads else np.nan,
    }
    if return_events:
        out["event_onsets"] = np.array([s for s, _ in events], dtype=tt.dtype)
        out["event_hit"] = np.array([i in hit_events for i in range(len(events))], dtype=bool)
    return out


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


# ---- operating point chosen on training data only ----------------------------------------------------------------

def offset_grid(train_change: np.ndarray, n: int = 61, k: float = 3.0) -> np.ndarray:
    """Candidate decision offsets in native units: n values in [-k, k] x the std of the training change y(t+h) - y(t).
    An alarm fires when forecast + offset > threshold, so a positive offset warns earlier and more often (persistence
    then alarms when y(t) is within `offset` below the threshold); a negative one alarms less."""
    s = float(np.std(train_change)) if len(train_change) else 0.0
    return np.linspace(-k * s, k * s, n) if s > 0 else np.zeros(1)


def choose_offset(t0, t_target, y_true, y_pred, threshold, horizon_min, budget_per_day, offsets, **kw):
    """Offset with the best recall subject to FA/day <= budget, scored on VALIDATION data (a training tail).

    A larger offset means more alarms. Ties in recall go to the smaller offset (fewer alarms). Without events in the
    validation data, the largest offset that keeps the budget is used (most sensitive within budget). If no offset keeps
    the budget, the one with the fewest false alarms per day is used (smallest offset on ties).
    Returns (offset, metrics at that offset)."""
    scored = [(float(o), evaluate_alerts(t0, t_target, y_true, y_pred, threshold, horizon_min, offset=o, **kw))
              for o in np.sort(np.asarray(offsets, dtype=float))]
    ok = [(o, r) for o, r in scored if r["fa_per_day"] <= budget_per_day]
    if not ok:
        return min(scored, key=lambda x: (x[1]["fa_per_day"], x[0]))
    if all(np.isnan(r["recall"]) for _, r in ok):
        return ok[-1]
    best = max(r["recall"] for _, r in ok if not np.isnan(r["recall"]))
    return [(o, r) for o, r in ok if r["recall"] == best][0]


# ---- events vs annotated activities ----------------------------------------------------------------------------

def event_onsets(times: np.ndarray, values: np.ndarray, threshold: float, min_sep_min: int = 15) -> np.ndarray:
    """Onset times of threshold events (same definition as evaluate_alerts)."""
    o = np.argsort(times)
    return np.array([s for s, _ in _runs(times[o], values[o] > threshold, np.timedelta64(min_sep_min, "m"))],
                    dtype=times.dtype)


def annotated(onsets: np.ndarray, ann_ts: np.ndarray, before_min: float = 60, after_min: float = 15,
              shift_min: float = 0) -> np.ndarray:
    """Boolean per onset: an annotation (shifted by shift_min) lies in [onset - before_min, onset + after_min]."""
    a = np.sort(np.asarray(ann_ts, dtype="datetime64[s]")) + np.timedelta64(int(shift_min * 60), "s")
    on = np.asarray(onsets, dtype="datetime64[s]")
    if len(a) == 0 or len(on) == 0:
        return np.zeros(len(on), dtype=bool)
    lo = np.searchsorted(a, on - np.timedelta64(int(before_min * 60), "s"), side="left")
    hi = np.searchsorted(a, on + np.timedelta64(int(after_min * 60), "s"), side="right")
    return hi > lo


def shift_scan(onsets, ann_ts, shifts_min, **kw) -> np.ndarray:
    """Share of onsets with an annotation nearby, for each clock shift applied to the annotations. A peak far from 0
    says the annotation clock and the sensor clock differ (e.g. local time vs UTC)."""
    return np.array([annotated(onsets, ann_ts, shift_min=s, **kw).mean() if len(onsets) else np.nan
                     for s in shifts_min])
