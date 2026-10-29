"""Statistics for the cold-start criterion and model comparison."""
from __future__ import annotations

import math

import numpy as np


def bootstrap_ci(values, n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0, groups=None):
    """Percentile bootstrap CI of the mean. If `groups` is given (e.g. split ids), resample whole groups
    (block bootstrap) so correlated rows from the same split are kept together."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if groups is None:
        means = [rng.choice(v, len(v)).mean() for _ in range(n_resamples)]
    else:
        g = np.asarray(groups)
        uniq = np.unique(g)
        buckets = [v[g == u] for u in uniq]
        means = []
        for _ in range(n_resamples):
            pick = rng.integers(0, len(uniq), len(uniq))
            means.append(np.concatenate([buckets[i] for i in pick]).mean())
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def geomean_ci(rel_values, groups=None, **kw):
    """Geometric mean of relative errors with a percentile bootstrap CI (bootstrap on log, back-transformed).
    The geometric mean is the recommended average for relative error measures (Hyndman & Koehler 2006): a single
    split with a near-zero persistence error cannot dominate it the way it dominates the arithmetic mean."""
    v = np.log(np.clip(np.asarray(rel_values, dtype=float), 1e-12, None))
    m, lo, hi = bootstrap_ci(v, groups=groups, **kw)
    return float(np.exp(m)), float(np.exp(lo)), float(np.exp(hi))


def meets_cold_start(rel_mae_values, groups=None, **kw) -> bool:
    """Cold-start criterion (proposal, revised 2026-10-28): the GEOMETRIC mean of rel_mae (MAE model / MAE persistence
    on the same test split) is < 1 and the upper bound of its 95% block-bootstrap CI is < 1. Pass split ids as
    `groups` so the seeds of one split stay together. Splits with too few test windows are filtered beforehand
    (scripts/summarize.py --min-test-frac)."""
    m, _, hi = geomean_ci(rel_mae_values, groups=groups, **kw)
    return m < 1 and hi < 1


def diebold_mariano(e1, e2, h: int = 1, power: int = 1):
    """DM test with Harvey-Leybourne-Newbold small-sample correction.
    e1, e2: forecast errors of two models on the same targets. h: horizon in steps.
    Returns (statistic, two-sided p-value); negative statistic means model 1 is better."""
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[: n - k] - dbar)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return float("nan"), float("nan")
    dm = dbar / math.sqrt(var)
    dm *= math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    # Student-t with n-1 df; use normal approximation via erf for large n, scipy if present
    try:
        from scipy.stats import t

        p = 2 * t.sf(abs(dm), n - 1)
    except Exception:
        p = math.erfc(abs(dm) / math.sqrt(2))
    return float(dm), float(p)
