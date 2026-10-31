import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("run_learning_curve",
                                               Path(__file__).parents[1] / "scripts/run_learning_curve.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def test_error_in_one_train_length_does_not_stop_the_others(capsys):
    def one_d(D):
        if D == 3:
            raise ValueError("Found array with 1 sample(s)")
        return [{"train_days": D, "split": s} for s in range(2)]

    rows, failed = runner.run_each([1, 3, 7], one_d)
    assert [r["train_days"] for r in rows] == [1, 1, 7, 7]  # the failed D leaves no partial rows
    assert list(failed) == [3] and "ValueError" in failed[3]
    assert "D=3 FAILED" in capsys.readouterr().out


_bspec = importlib.util.spec_from_file_location("run_batch", Path(__file__).parents[1] / "scripts/run_batch.py")
batch = importlib.util.module_from_spec(_bspec)
_bspec.loader.exec_module(batch)


def test_native_cadence_from_time_steps():
    """SYNTHETIC time stamps (code test only)."""
    import pandas as pd

    t0 = pd.Timestamp("2024-01-01")
    one_min = t0 + pd.to_timedelta(range(1000), "min")
    assert batch.native_cadence(one_min) == 5
    # 5-min logger with 9 % of 10-min steps (Urbino's pattern): a 5-min grid would split there
    steps = [10 if i % 11 == 0 else 5 for i in range(1000)]
    assert batch.native_cadence(t0 + pd.to_timedelta(pd.Series(steps).cumsum(), "min")) == 10
    assert batch.native_cadence(t0 + pd.to_timedelta(range(0, 30000, 30), "min")) is None
    assert batch.HORIZONS[10] == [30, 60]


def test_batch3_runs_linear_models_and_policy_in_its_own_folder(tmp_path, monkeypatch):
    assert batch.BATCH_MODELS[3] == ["ridge", "huber", "policy"] and batch.BATCH_OUT[3] != batch.BATCH_OUT[1]
    seen = {}
    monkeypatch.setattr(batch.subprocess, "run", lambda cmd, **k: seen.setdefault("cmd", cmd) and type("R", (), {"returncode": 0}))
    (tmp_path / "logs").mkdir()
    batch.run(("dalton", "p", "H1_41", "eco2", 5, ["1", "3"], 15), 5, tmp_path, batch.BATCH_MODELS[3])
    cmd = seen["cmd"]
    assert cmd[cmd.index("--models") + 1:cmd.index("--parallel")] == ["ridge", "huber", "policy"]


def test_batch4_passes_weekly_flag(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(batch.subprocess, "run", lambda cmd, **k: seen.setdefault("cmd", cmd) and type("R", (), {"returncode": 0}))
    (tmp_path / "logs").mkdir()
    batch.run(("urbino", "p", "olivetti", "eco2", 10, ["1", "7"], 30), 5, tmp_path, batch.BATCH_MODELS[4],
              batch.BATCH_EXTRA[4])
    assert "--weekly" in seen["cmd"] and batch.BATCH_OUT[4] == "results/runs_batch4"
