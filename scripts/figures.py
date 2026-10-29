"""Paper figures from summarize.py outputs.

Main figure: share of devices that meet the cold-start criterion at each training length D, among the devices tested
at that D; one line per model, one panel per forecast horizon.

  python scripts/summarize.py "results/runs/dalton_*_eco2_*.csv" --out results/summary_dalton_eco2
  python scripts/figures.py results/summary_dalton_eco2_fraction.csv --out results/fig_fraction_dalton_eco2
  -> .png (preview) and .pdf (paper)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Fixed model -> color/marker mapping (color follows the model, never its rank). Categorical slots 1-6 of the
# reference palette (validated for adjacent pairs in light mode); the marker is a second cue for print and CVD.
STYLE = {
    "huber": ("#2a78d6", "o"),
    "ridge": ("#eb6834", "s"),
    "policy": ("#1baf7a", "D"),
    "lightgbm": ("#eda100", "^"),
    "random_forest": ("#e87ba4", "v"),
    "gru": ("#008300", "P"),
}
LABEL = {"huber": "Huber", "ridge": "Ridge", "policy": "Policy", "lightgbm": "LightGBM",
         "random_forest": "Random forest", "gru": "GRU"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def fraction_figure(frac: pd.DataFrame, title: str | None = None):
    """frac: summarize.py *_fraction.csv rows (one dataset and target). Returns the matplotlib Figure."""
    frac = frac[frac["model"].isin(STYLE)]
    horizons = sorted(frac["horizon_min"].unique())
    days = sorted(frac["train_days"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(max(4.2, 3.4 * len(horizons)), 3.4), sharey=True,
                             squeeze=False)
    ymax = min(100, 10 * (int(frac["frac_meeting"].max() * 100) // 10 + 1))
    for ax, h in zip(axes[0], horizons):
        sub = frac[frac["horizon_min"] == h]
        for model, (color, marker) in STYLE.items():
            m = sub[sub["model"] == model].sort_values("train_days")
            if len(m):
                ax.plot(m["train_days"], 100 * m["frac_meeting"], color=color, marker=marker, lw=1.6, ms=5,
                        mec="white", mew=0.8, label=LABEL[model], zorder=3)
        n = sub.groupby("train_days")["n_devices"].max()
        ax.set_xscale("log")
        ax.set_xticks(days)
        ax.set_xticklabels([f"{d:g}\nn={n.get(d, 0)}" for d in days], fontsize=7, color=MUTED)
        ax.minorticks_off()
        ax.set_ylim(0, ymax)
        ax.set_title(f"{h:g}-min horizon", fontsize=9.5, color=INK)
        ax.set_xlabel("Training days D  (n = devices tested)", fontsize=8.5, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.tick_params(axis="y", labelsize=8, colors=MUTED, length=0)
        ax.tick_params(axis="x", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
    axes[0][0].set_ylabel("Devices meeting the criterion (%)", fontsize=8.5, color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    ncol = len(labels) if len(horizons) > 1 else 3  # one panel is too narrow for a single legend row
    rows = -(-len(labels) // ncol)
    head = 0.07 if title else 0.0
    fig.tight_layout(rect=(0, 0, 1, 1 - head - 0.065 * rows))
    fig.legend(handles, labels, loc="upper center", ncol=ncol, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1 - head), handlelength=2.2, columnspacing=1.2)
    if title:
        fig.suptitle(title, y=1.0, fontsize=10, color=INK)
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("fraction_csv", help="summarize.py *_fraction.csv")
    p.add_argument("--out", required=True, help="output path without extension")
    p.add_argument("--title")
    a = p.parse_args()
    frac = pd.read_csv(a.fraction_csv)
    fig = fraction_figure(frac, a.title)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {a.out}.png and {a.out}.pdf")


if __name__ == "__main__":
    main()
