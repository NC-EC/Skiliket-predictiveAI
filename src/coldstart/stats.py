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


def meets_cold_start(mase_values, groups=None, **kw) -> bool:
    """Criterion from the proposal: mean MASE < 1 and the upper 95% CI bound < 1."""
    m, _, hi = bootstrap_ci(mase_values, groups=groups, **kw)
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
