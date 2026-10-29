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
