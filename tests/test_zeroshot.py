"""Zero-shot runner with a FAKE pipeline and SYNTHETIC data (code tests only, not results)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from coldstart import zeroshot as zs


class FakeBolt:
    """Forecasts the last context value plus one per step; returns (batch, steps, quantiles) like Chronos-Bolt."""
    model = torch.nn.Linear(3, 2)

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        last = torch.stack([x[-1] for x in inputs])
        q = last[:, None] + torch.arange(1, prediction_length + 1, dtype=torch.float32)[None, :]
        return q[:, :, None], q


class FakeChronos2(FakeBolt):
    """Same forecasts, Chronos-2 layout: a list of (variates, steps, quantiles) tensors; inputs may be dicts."""
    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        out = []
        for x in inputs:
            last = float(np.asarray(x["target"] if isinstance(x, dict) else x)[-1])
            q = torch.tensor([last + s for s in range(1, prediction_length + 1)], dtype=torch.float32)
            out.append(q[None, :, None])
        return out, [o[..., 0] for o in out]


def _grid(days=16, step=5):
    t = pd.date_range("2025-01-01", periods=days * 1440 // step, freq=f"{step}min")
    seg = (np.arange(len(t)) >= len(t) // 2).astype(int)  # two segments
    rng = np.random.default_rng(0)
    return pd.DataFrame({"ts": t, "segment": seg, "eco2": 500 + np.cumsum(rng.normal(0, 3, len(t))),
                         "temp_c": 22 + rng.normal(0, 0.1, len(t))})


def test_contexts_stop_at_segment_start():
    g = _grid()
    mid = g["ts"].iloc[len(g) // 2 + 3]  # 4th point of the second segment
    c = zs.contexts(g, "eco2", np.array([mid.to_datetime64()]), 288)[0]
    assert len(c) == 4 and c[-1] == g["eco2"].iloc[len(g) // 2 + 3]
    c2 = zs.contexts(g, "eco2", np.array([mid.to_datetime64()]), 288, ["temp_c"])[0]
    assert set(c2) == {"target", "past_covariates"} and len(c2["past_covariates"]["temp_c"]) == 4


def test_median_forecast_handles_both_layouts():
    ctx = [np.array([1.0, 2.0]), np.array([5.0])]
    for pipe in (FakeBolt(), FakeChronos2()):
        f = zs.median_forecast(pipe, ctx, 3, batch_size=1)
        assert f.shape == (2, 3) and np.allclose(f, [[3, 4, 5], [6, 7, 8]])
    assert zs.n_params(FakeBolt()) == 8


def test_run_zeroshot_end_to_end_and_summarize(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("rz", Path(__file__).parents[1] / "scripts/run_zeroshot.py")
    rz = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rz)
    g = _grid(days=20)
    monkeypatch.setattr(rz, "load_grid", lambda *a, **k: (g, ["eco2", "temp_c"], "synthetic"))
    monkeypatch.setattr(rz, "load_pipeline", lambda m: FakeBolt())
    out = tmp_path / "z.csv"
    rz.main(["--dataset", "syn", "--path", ".", "--devices", "X1", "--horizons", "15", "30", "--days", "1", "3",
             "--context-hours", "1", "6", "--out", str(out)])
    df = pd.read_csv(out)
    assert set(df["model"]) == {"persistence", "chronos_bolt_tiny_ctx1h", "chronos_bolt_tiny_ctx6h"}
    assert (df["train_days"] == 0).all() and set(df["horizon_min"]) == {15, 30}
    # the fake forecasts last + k, a known bias, so it must differ from persistence and be finite
    z = df[df.model == "chronos_bolt_tiny_ctx1h"]
    assert np.isfinite(z["rel_mae"]).all() and (z["rel_mae"] != 1).all() and (z["n_params"] == 8).all()
    # same test days as the learning curve's largest feasible D (3 days here): first test day starts after day 3
    assert pd.to_datetime(df["test_start"]).min() >= pd.Timestamp("2025-01-04")
    # resumable: a second run reuses the part
    rz.main(["--dataset", "syn", "--path", ".", "--devices", "X1", "--horizons", "15", "30", "--days", "1", "3",
             "--context-hours", "1", "6", "--out", str(out)])
    assert len(pd.read_csv(out)) == len(df)
    s_spec = importlib.util.spec_from_file_location("summ", Path(__file__).parents[1] / "scripts/summarize.py")
    summ = importlib.util.module_from_spec(s_spec)
    s_spec.loader.exec_module(summ)
    cells = summ.summarize_cells(df, n_resamples=50)
    assert {"chronos_bolt_tiny_ctx1h", "persistence"} <= set(cells["model"])
