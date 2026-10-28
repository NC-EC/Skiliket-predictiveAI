"""PIPELINE SMOKE TEST on the SKILIKET file. NOT a research result (1 device, 6.8 days, 1 seed).
Checks that adapter -> grid -> windows -> rolling origin -> baselines/Ridge -> MASE -> cost all run end to end.
Usage: python scripts/smoke_skiliket.py data/raw/skiliket_zenodo_17398551.xlsx
"""
import sys

import numpy as np
from sklearn.linear_model import Ridge

from coldstart import evaluate as ev
from coldstart.adapters import skiliket
from coldstart.cost import measure_fit, measure_inference

df, meta, rep = skiliket.load(sys.argv[1])
cad, h_min, n_lags = 5, 15, 6
chs = ["eco2", "tvoc", "temp_c", "rh_pct"]
grid = ev.to_grid(df, cad, chs)
w = ev.make_windows(grid, "eco2", chs, n_lags, h_min // cad)
purge = h_min + n_lags * cad
print(f"grid rows={len(grid)} windows={len(w.y)} cadence={cad}min horizon={h_min}min")

for train_days in (1, 2, 3):
    res = {"persist": [], "ma": [], "ridge": []}
    costs = []
    for tr, te in ev.rolling_origin(w, train_days, 1.0, purge):
        res["persist"].append(ev.evaluate_split(w, tr, te, lambda a, b: ev.persistence(w, b))["mase"])
        res["ma"].append(ev.evaluate_split(w, tr, te, lambda a, b: ev.moving_average(w, b, len(chs)))["mase"])

        def fit_pred(a, b):
            m, c = measure_fit(lambda: Ridge(alpha=1.0).fit(w.X[a], w.y[a] - w.y0[a]))
            costs.append((c, m, w.X[b]))
            return w.y0[b] + m.predict(w.X[b])

        res["ridge"].append(ev.evaluate_split(w, tr, te, fit_pred)["mase"])
    if res["persist"]:
        c, m, Xb = costs[-1]
        print(f"train={train_days}d splits={len(res['persist'])} MASE persist={np.mean(res['persist']):.2f} "
              f"MA={np.mean(res['ma']):.2f} Ridge={np.mean(res['ridge']):.2f} | ridge fit {c.wall_s*1000:.1f} ms, "
              f"{c.model_bytes} B, infer {measure_inference(m, Xb):.2f} ms")
