import importlib.util
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("matplotlib")
_spec = importlib.util.spec_from_file_location("figures", Path(__file__).parents[1] / "scripts/figures.py")
figures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(figures)


def test_one_panel_per_horizon_one_line_per_model(tmp_path):
    """SYNTHETIC fraction table (code test only)."""
    rows = [{"dataset": "s", "target": "eco2", "horizon_min": h, "model": m, "train_days": d,
             "n_devices": 10, "n_meeting": i, "frac_meeting": i / 10}
            for h in (15, 30, 60) for m in ("huber", "ridge", "persistence") for i, d in enumerate((1, 3, 7))]
    fig = figures.fraction_figure(pd.DataFrame(rows), "synthetic")
    axes = fig.get_axes()
    assert len(axes) == 3 and [a.get_title() for a in axes] == ["15-min horizon", "30-min horizon", "60-min horizon"]
    assert all(len(a.get_lines()) == 2 for a in axes)  # persistence is not plotted (always 0 by definition)
    assert axes[0].get_lines()[0].get_color() == figures.STYLE["huber"][0]  # color follows the model
    fig.savefig(tmp_path / "f.png")
    assert (tmp_path / "f.png").stat().st_size > 0


def test_paper_figures_draw_each_series(tmp_path):
    """SYNTHETIC summaries (code test only)."""
    frac = pd.DataFrame([{"horizon_min": h, "model": m, "train_days": d, "n_devices": 10, "frac_meeting": i / 10}
                         for h in (15, 30, 60) for m in figures.LC_MODELS for i, d in enumerate((1, 3, 7))])
    fig = figures.learning_curve_figure(frac, frac)
    assert len(fig.get_axes()) == 3 and all(len(a.get_lines()) == 8 for a in fig.get_axes())  # 4 models x 2 targets
    assert {ln.get_linestyle() for ln in fig.get_axes()[0].get_lines()} == {"-", ":"}

    dt = pd.DataFrame([{"device": "all", "model": "huber", "horizon_min": h, "train_days": d, "day_type": k,
                        "geo_rel_mae": 1.2, "geo_ci_lo": 1.1, "geo_ci_hi": 1.3}
                       for h in (30, 60) for d in (1, 7, 60) for k in figures.DAYTYPE])
    assert len(figures.urbino_daytype_figure(dt).get_axes()) == 2

    ov = pd.DataFrame([{"train_days": 14, "mode": "tuned", "alert_q": 90, "model": m, "budget": b,
                        "median_fa_per_day": b * 0.8, "median_recall": 0.3 + b / 10}
                       for m in figures.ALERT_MODELS for b in (0.5, 1, 2)])
    fig = figures.alerts_tradeoff_figure(ov)
    assert len(fig.get_axes()[0].get_lines()) == len(figures.ALERT_MODELS)
    fig.savefig(tmp_path / "a.svg")
    assert (tmp_path / "a.svg").stat().st_size > 0
