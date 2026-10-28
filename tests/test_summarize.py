import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location("summarize", Path(__file__).parents[1] / "scripts/summarize.py")
summarize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarize)


def synthetic_runs(tmp_path):
    """Labeled SYNTHETIC runner rows, only to test the summary code (not results).
    Model 'good' loses at D=1, wins clearly from D=3; 'flat' hovers around 1 and never meets the criterion."""
    rng = np.random.default_rng(0)
    rows = []
    for D, good_level in [(1, 1.10), (3, 0.80), (7, 0.70)]:
        for split in range(12):
            base = {"dataset": "synth", "device": "dev1", "target": "eco2", "horizon_min": 15, "train_days": D,
                    "split": split}
            rows.append({**base, "model": "persistence", "seed": 0, "rel_mae": 1.0})
            for seed in range(3):
                rows.append({**base, "model": "good", "seed": seed, "rel_mae": good_level + rng.normal(0, 0.02),
                             "wall_s": 0.1 * D, "peak_py_mb": 1.0 + seed, "parallel": 5})
            rows.append({**base, "model": "flat", "seed": 0, "rel_mae": 1.0 + (0.1 if split % 2 else -0.1)})
    path = tmp_path / "synthetic_run.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_cells_and_d_star(tmp_path):
    cells = summarize.summarize_cells(summarize.load([str(synthetic_runs(tmp_path))]), n_resamples=200)
    good = cells[cells.model == "good"].set_index("train_days")
    assert good["n_splits"].tolist() == [12, 12, 12]
    assert good.loc[1, "pct_splits_won"] == 0 and good.loc[3, "pct_splits_won"] == 100
    assert not good.loc[1, "meets_cold_start"] and good.loc[3, "meets_cold_start"]
    assert good.loc[3, "ci_lo"] <= good.loc[3, "mean_rel_mae"] <= good.loc[3, "ci_hi"]
    assert good.loc[7, "train_s"] == pytest.approx(0.7) and good.loc[7, "peak_py_mb"] == 2.0 and good.loc[7, "parallel"] == 5
    flat = cells[cells.model == "flat"]
    assert flat["pct_splits_won"].eq(50).all() and not flat["meets_cold_start"].any()

    ds = summarize.d_star(cells).set_index("model")
    assert ds.loc["good", "d_star"] == 3
    assert np.isnan(ds.loc["flat", "d_star"]) and np.isnan(ds.loc["persistence", "d_star"])
    assert ds.loc["good", "days_tested"] == "1 3 7"


def test_device_from_file_name_for_old_csvs(tmp_path):
    path = synthetic_runs(tmp_path)
    df = pd.read_csv(path).drop(columns="device")
    old = tmp_path / "smoke_dalton_X1_11_h15.csv"
    df.to_csv(old, index=False)
    assert set(summarize.load([str(old)])["device"]) == {"smoke_dalton_X1_11_h15"}
