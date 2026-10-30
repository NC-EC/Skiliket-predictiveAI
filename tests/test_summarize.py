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


def test_short_split_dropped_and_geometric_mean_is_robust(tmp_path):
    """SYNTHETIC rows (code test only): one 10-window split with a near-zero persistence error gives rel_mae 33."""
    rows = []
    for split in range(12):
        n = 10 if split == 0 else 288
        rel = 33.0 if split == 0 else 0.8
        base = {"dataset": "synth", "device": "d", "target": "eco2", "horizon_min": 15, "train_days": 3,
                "split": split, "n_test": n}
        rows.append({**base, "model": "persistence", "seed": 0, "rel_mae": 1.0, "mae": 10.0})
        rows.append({**base, "model": "m", "seed": 0, "rel_mae": rel, "mae": 10.0 * rel})
    path = tmp_path / "synth.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    df = summarize.load([str(path)])

    kept = summarize.summarize_cells(df, n_resamples=200).set_index("model").loc["m"]
    assert kept["n_splits"] == 11 and kept["meets_cold_start"]
    assert kept["geo_rel_mae"] == pytest.approx(0.8) and kept["pooled_rel_mae"] == pytest.approx(0.8)

    raw = summarize.summarize_cells(df, n_resamples=200, min_test_frac=0).set_index("model").loc["m"]
    assert raw["n_splits"] == 12 and raw["mean_rel_mae"] > 3      # the arithmetic mean is wrecked by one split
    assert raw["geo_rel_mae"] < 1.1                                # the geometric mean much less so


def _cells(spec):
    """SYNTHETIC cell table (code test only): {device: {D: meets}} for one model."""
    return pd.DataFrame([{"dataset": "s", "device": dev, "target": "eco2", "horizon_min": 15, "model": "m",
                          "train_days": D, "meets_cold_start": ok}
                         for dev, by_d in spec.items() for D, ok in by_d.items()])


def test_stable_d_star_status_and_fraction():
    cells = _cells({
        "stable": {1: False, 3: True, 7: True, 14: True},
        "flicker": {1: True, 3: False, 7: True, 14: True},    # first = 1, stable = 7
        "lost": {1: False, 3: True, 7: True, 14: False},      # tested at 14 and fails there: never
        "short_ok": {1: False, 3: True},                      # stable within what it has
        "short_no": {1: False, 3: False},                     # too few days for 14: untestable
    })
    ds = summarize.d_star(cells).set_index("device")
    assert ds.loc["stable", ["d_star", "d_star_status"]].tolist() == [3, "reached"]
    assert ds.loc["flicker", "d_star"] == 7 and ds.loc["flicker", "d_star_first"] == 1
    assert np.isnan(ds.loc["lost", "d_star"]) and ds.loc["lost", "d_star_status"] == "never"
    assert ds.loc["lost", "d_star_first"] == 3
    assert ds.loc["short_ok", "d_star"] == 3 and ds.loc["short_ok", "d_star_status"] == "unconfirmed"
    assert ds.loc["short_no", "d_star_status"] == "untestable" and ds.loc["short_no", "max_days_tested"] == 3

    fr = summarize.fraction_meeting(cells).set_index("train_days")
    assert fr.loc[3, ["n_devices", "n_meeting"]].tolist() == [5, 3]
    assert fr.loc[14, "n_devices"] == 3 and fr.loc[14, "frac_meeting"] == pytest.approx(2 / 3)


def test_policy_recomputed_from_validation_scores():
    """SYNTHETIC runner rows (code test only). Split 0: huber validates best -> huber's test row.
    Split 1: no candidate validates below 1 -> persistence. LightGBM validates best of all but is not a candidate."""
    rows = []
    for split, (cv_r, cv_h) in enumerate([(0.9, 0.8), (1.1, np.nan)]):
        b = {"dataset": "s", "device": "d", "target": "eco2", "horizon_min": 15, "train_days": 3, "split": split,
             "seed": 0}
        rows += [{**b, "model": "persistence", "rel_mae": 1.0, "mae": 10.0},
                 {**b, "model": "ridge", "rel_mae": 0.95, "mae": 9.5, "cv_rel_mae": cv_r, "wall_s": 1.0},
                 {**b, "model": "huber", "rel_mae": 0.85, "mae": 8.5, "cv_rel_mae": cv_h, "wall_s": 2.0},
                 {**b, "model": "lightgbm", "rel_mae": 1.3, "mae": 13.0, "cv_rel_mae": 0.5, "wall_s": 0.1},
                 {**b, "model": "policy", "rel_mae": 1.3, "mae": 13.0, "policy_choice": "lightgbm"}]
    out = summarize.recompute_policy(pd.DataFrame(rows))
    pol = out[out.model == "policy"].set_index("split")
    assert len(pol) == 2
    assert pol.loc[0, "policy_choice"] == "huber" and pol.loc[0, "rel_mae"] == 0.85 and pol.loc[0, "wall_s"] == 3.0
    assert pol.loc[1, "policy_choice"] == "persistence" and pol.loc[1, "rel_mae"] == 1.0
    # CSVs without validation scores keep the runner's policy rows
    old = pd.DataFrame(rows).drop(columns="cv_rel_mae")
    assert summarize.recompute_policy(old).equals(old)


def test_single_d_is_insufficient_and_largest_d_is_unconfirmed():
    cells = _cells({
        "one_d_meets": {1: True},                    # one D tested: stability cannot be judged
        "one_d_fails": {1: False},
        "last_only": {1: False, 3: False, 7: True},  # meets only at its largest D: unconfirmed
        "full": {1: False, 3: True, 7: True, 14: True},
    })
    ds = summarize.d_star(cells).set_index("device")
    for dev in ("one_d_meets", "one_d_fails"):
        assert ds.loc[dev, "d_star_status"] == "insufficient" and np.isnan(ds.loc[dev, "d_star"])
    assert ds.loc["one_d_meets", "d_star_first"] == 1
    assert ds.loc["last_only", ["d_star", "d_star_status"]].tolist() == [7, "unconfirmed"]
    assert ds.loc["full", ["d_star", "d_star_status"]].tolist() == [3, "reached"]
    # meeting at 14 = the design maximum with a smaller D failing: still unconfirmed, not reached
    edge = summarize.d_star(_cells({"a": {7: False, 14: True}, "b": {1: True, 14: True}})).set_index("device")
    assert edge.loc["a", "d_star_status"] == "unconfirmed" and edge.loc["b", "d_star_status"] == "reached"


def test_policy_recomputed_only_for_files_with_validation_scores(tmp_path):
    """SYNTHETIC rows (code test only): a new CSV (cv_rel_mae) loaded with an old one (no such column) must not
    turn the old file's policy rows into persistence."""
    def rows(device, cv):
        b = {"dataset": "s", "device": device, "target": "eco2", "horizon_min": 15, "train_days": 3, "split": 0,
             "seed": 0}
        r = [{**b, "model": "persistence", "rel_mae": 1.0}, {**b, "model": "ridge", "rel_mae": 0.9},
             {**b, "model": "huber", "rel_mae": 0.8}, {**b, "model": "policy", "rel_mae": 0.7,
                                                      "policy_choice": "lightgbm"}]
        if cv:
            for x in r[1:3]:
                x["cv_rel_mae"] = 0.95 if x["model"] == "ridge" else 0.9
        return pd.DataFrame(r)

    rows("new", True).to_csv(tmp_path / "new.csv", index=False)
    rows("old", False).to_csv(tmp_path / "old.csv", index=False)
    df = summarize.load([str(tmp_path / "new.csv"), str(tmp_path / "old.csv")])
    pol = summarize.recompute_policy(df).query("model == 'policy'").set_index("device")
    assert len(pol) == 2
    assert pol.loc["old", "policy_choice"] == "lightgbm" and pol.loc["old", "rel_mae"] == 0.7
    assert pol.loc["new", "policy_choice"] == "huber" and pol.loc["new", "rel_mae"] == 0.8


def test_policy_threshold_is_a_parameter_with_default_one():
    """SYNTHETIC rows (code test only): huber validates at 0.97, below 1 but not below 0.95."""
    b = {"dataset": "s", "device": "d", "target": "eco2", "horizon_min": 15, "train_days": 3, "split": 0, "seed": 0}
    df = pd.DataFrame([{**b, "model": "persistence", "rel_mae": 1.0},
                       {**b, "model": "ridge", "rel_mae": 0.9, "cv_rel_mae": 1.02},
                       {**b, "model": "huber", "rel_mae": 0.8, "cv_rel_mae": 0.97}])
    choice = lambda **kw: summarize.recompute_policy(df, **kw).query("model == 'policy'")["policy_choice"].item()
    assert choice() == "huber" and choice(threshold=0.95) == "persistence"
