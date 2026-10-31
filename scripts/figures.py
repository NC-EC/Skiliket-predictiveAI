"""Paper figures from summarize.py outputs.

Main figure: share of devices that meet the cold-start criterion at each training length D, among the devices tested
at that D; one line per model, one panel per forecast horizon.

  python scripts/summarize.py "results/runs/dalton_*_eco2_*.csv" --out results/summary_dalton_eco2
  python scripts/figures.py results/summary_dalton_eco2_fraction.csv --out results/fig_fraction_dalton_eco2
  -> .png (preview) and .pdf (paper)

--paper builds the final figures and tables from the committed summaries only (no model is run):
  fig_learning_curve   DALTON share of devices meeting the criterion vs D, eCO2 (solid; huber/ridge/lightgbm from
                       batch 1, the corrected policy from batch 3) and TVOC (dotted; batch 2), one panel per horizon;
  fig_urbino_daytype   Urbino huber geometric rel_mae with 95 % CI vs D, weekday vs weekend/holiday (calendar split);
  fig_alerts_tradeoff  DALTON eCO2 h30 alerts at D=14: median recall vs median false alarms/day, budgets 0.5/1/2
                       (q90 threshold, tuned offset);
  tables_paper.md      Table 2 (stable D*, policy rows from batch 3) and the edge cost table (results/edge/, D=7).
No titles inside the figures (they go in the caption). PNG at 300 dpi and SVG.

  python scripts/figures.py --paper --out-dir results/figures
"""
from __future__ import annotations

import argparse
import glob
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
PERSISTENCE = ("#52514e", "X")  # the baseline: neutral ink, not a categorical slot
PAPER_RC = {"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
            "legend.fontsize": 10, "svg.fonttype": "none", "font.family": "DejaVu Sans"}


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


def _clean(ax):
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)


def _panel_label(ax, text):
    ax.text(0.0, 1.02, text, transform=ax.transAxes, ha="left", va="bottom", color=INK, fontsize=10.5)


LC_MODELS = ["huber", "ridge", "lightgbm", "policy"]


def learning_curve_figure(eco2: pd.DataFrame, tvoc: pd.DataFrame):
    """eco2 / tvoc: *_fraction.csv rows (one dataset) with the models to draw. Solid eCO2, dotted TVOC."""
    from matplotlib.lines import Line2D

    horizons = sorted(eco2["horizon_min"].unique())
    days = sorted(eco2["train_days"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(7.2, 3.0), sharey=True, squeeze=False)
    for ax, h in zip(axes[0], horizons):
        for frac, ls, ms in ((eco2, "-", 5), (tvoc, ":", 4)):
            sub = frac[frac["horizon_min"] == h]
            for model in LC_MODELS:
                color, marker = STYLE[model]
                m = sub[sub["model"] == model].sort_values("train_days")
                ax.plot(m["train_days"], 100 * m["frac_meeting"], color=color, marker=marker, ls=ls, lw=1.6, ms=ms,
                        mec="white", mew=0.7, zorder=3 if ls == "-" else 2)
        n_e = eco2[eco2["horizon_min"] == h].groupby("train_days")["n_devices"].max()
        n_t = tvoc[tvoc["horizon_min"] == h].groupby("train_days")["n_devices"].max()
        ax.set_xscale("log")
        ax.set_xticks(days)
        ax.set_xticklabels([f"{d:g}\n{n_e.get(d, 0)}" if n_e.get(d) == n_t.get(d)
                            else f"{d:g}\n{n_e.get(d, 0)}/{n_t.get(d, 0)}" for d in days])
        ax.minorticks_off()
        ax.set_ylim(0, 60)
        _clean(ax)
        _panel_label(ax, f"{h:g}-min horizon")
    axes[0][0].set_ylabel("Devices meeting criterion (%)")
    axes[0][len(horizons) // 2].set_xlabel("Training days D (second row: devices tested)")
    handles = [Line2D([], [], color=STYLE[m][0], marker=STYLE[m][1], lw=1.6, ms=5, mec="white", label=LABEL[m])
               for m in LC_MODELS]
    handles += [Line2D([], [], color=MUTED, ls="-", lw=1.6, label="eCO$_2$"),
                Line2D([], [], color=MUTED, ls=":", lw=1.6, label="TVOC")]
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.0, columnspacing=1.0)
    return fig


DAYTYPE = {"weekday": ("Weekday", INK, "o", "-"), "weekend_holiday": ("Weekend / holiday", "#8a8984", "s", "--")}


def urbino_daytype_figure(dt: pd.DataFrame, model: str = "huber"):
    """dt: check_urbino.py *_daytype_calendar.csv; rows of device 'all'."""
    dt = dt[(dt["device"] == "all") & (dt["model"] == model)]
    horizons = sorted(dt["horizon_min"].unique())
    days = sorted(dt["train_days"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(7.2, 3.0), sharey=True, squeeze=False)
    for ax, h in zip(axes[0], horizons):
        for i, (kind, (label, color, marker, ls)) in enumerate(DAYTYPE.items()):
            g = dt[(dt["horizon_min"] == h) & (dt["day_type"] == kind)].sort_values("train_days")
            x = g["train_days"] * (0.96 if i == 0 else 1.04)  # dodge the CIs on the log axis
            err = [g["geo_rel_mae"] - g["geo_ci_lo"], g["geo_ci_hi"] - g["geo_rel_mae"]]
            ax.errorbar(x, g["geo_rel_mae"], yerr=err, color=color, marker=marker, ls=ls, lw=1.6, ms=5, mec="white",
                        mew=0.7, capsize=2.5, elinewidth=1.0, label=label, zorder=3)
        ax.axhline(1, color=MUTED, lw=1.0, zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(days)
        ax.set_xticklabels([f"{d:g}" for d in days])
        ax.set_yticks([0.8, 1, 1.5, 2, 3])
        ax.set_yticklabels(["0.8", "1", "1.5", "2", "3"])
        ax.minorticks_off()
        _clean(ax)
        _panel_label(ax, f"{h:g}-min horizon")
        ax.set_xlabel("Training days D")
    axes[0][0].set_ylabel("Rel. MAE (geometric mean, 95% CI)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    return fig


ALERT_MODELS = ["persistence", "ridge", "huber", "lightgbm", "policy"]


def alerts_tradeoff_figure(ov: pd.DataFrame, D: float = 14, q: float = 90):
    """ov: summarize_alerts.py *_overall.csv. One point per model and false-alarm budget (tuned offset)."""
    ov = ov[(ov["train_days"] == D) & (ov["mode"] == "tuned") & (ov["alert_q"] == q)]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for model in ALERT_MODELS:
        color, marker = PERSISTENCE if model == "persistence" else STYLE[model]
        g = ov[ov["model"] == model].sort_values("budget")
        ax.plot(g["median_fa_per_day"], g["median_recall"], color=color, marker=marker, lw=1.4, ms=6.5,
                mec="white", mew=0.8, ls="--" if model == "persistence" else "-",
                label="Persistence" if model == "persistence" else LABEL[model], zorder=3)
        if model == "persistence":  # budgets named once, on the baseline
            for _, r in g.iterrows():
                ax.annotate(f"budget {r['budget']:g}", (r["median_fa_per_day"], r["median_recall"]), xytext=(-7, 5),
                            textcoords="offset points", ha="right", fontsize=9, color=MUTED)
    ax.set_xlabel("False alarms per day (median)")
    ax.set_ylabel("Recall (median)")
    ax.set_xlim(left=0)
    lo, hi = ov["median_recall"].min(), ov["median_recall"].max()
    ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
    _clean(ax)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def _status_table(ds: pd.DataFrame) -> list[str]:
    out = ["| Horizon | Model | Reached | Unconfirmed | Never | Untestable | Insufficient | Median D* (reached) [IQR] |",
           "|---|---|---|---|---|---|---|---|"]
    order = {m: i for i, m in enumerate(["huber", "ridge", "lightgbm", "random_forest", "gru", "policy"])}
    ds = ds.assign(_o=ds["model"].map(order)).sort_values(["horizon_min", "_o"])
    for (h, model), g in ds.groupby(["horizon_min", "model"], sort=False):
        c = g["d_star_status"].value_counts()
        r = g.loc[g["d_star_status"] == "reached", "d_star"]
        med = f"{r.median():g} [{r.quantile(.25):g}-{r.quantile(.75):g}]" if len(r) else "-"
        counts = " | ".join(str(int(c.get(k, 0))) for k in ("reached", "unconfirmed", "never", "untestable",
                                                            "insufficient"))
        out.append(f"| {h:g} min | {LABEL.get(model, model)} | {counts} | {med} |")
    return out


def cost_table(edge: pd.DataFrame, D: float = 7) -> list[str]:
    e = edge[(edge["train_days"] == D) & edge["wall_s"].notna()]
    cfg = e.groupby(["config", "model"])["wall_s"].median().unstack(0)
    med = e.groupby("model")[["wall_s", "peak_py_mb", "infer_ms", "model_bytes"]].median()
    out = ["| Model | Train (s) | Train, range over configs (s) | Peak Python memory (MB) | Inference (ms/window) "
           "| Model size (kB) |", "|---|---|---|---|---|---|"]
    for model, r in med.sort_values("wall_s").iterrows():
        out.append(f"| {LABEL.get(model, model)} | {r['wall_s']:.2f} | {cfg.loc[model].min():.2f}-"
                   f"{cfg.loc[model].max():.2f} | {r['peak_py_mb']:.2f} | {r['infer_ms']:.3f} | "
                   f"{r['model_bytes'] / 1000:.1f} |")
    return out


def paper_tables(root: Path) -> str:
    b1 = pd.read_csv(root / "summary_batch1_dalton_dstar.csv")
    b3 = pd.read_csv(root / "summary_batch3_dalton_dstar.csv")
    tv = pd.read_csv(root / "summary_batch2_dalton_tvoc_dstar.csv")
    keep = ["huber", "ridge", "lightgbm", "random_forest", "gru"]
    eco2 = pd.concat([b1[b1["model"].isin(keep)], b3[b3["model"] == "policy"]])
    tvoc = tv[tv["model"].isin(keep + ["policy"])]
    frames = []
    for f in sorted(glob.glob(str(root / "edge" / "*.csv"))):
        d = pd.read_csv(f)
        d["config"] = Path(f).stem.split("_", 2)[-1]
        frames.append(d)
    edge = pd.concat(frames, ignore_index=True)
    lines = [
        "# Paper tables", "",
        "Generated by `scripts/figures.py --paper` from committed summaries; do not edit by hand.", "",
        "## Table 2. Stable D* per device, DALTON", "",
        "Stable D* = smallest training length D that meets the cold-start criterion (geometric mean rel. MAE < 1 and "
        "upper 95% block-bootstrap bound < 1) at that D and at every larger D tested. Counts are devices. Policy rows: "
        "corrected policy (ridge + huber candidates, chosen on cv_rel_mae; batch 3). Other eCO2 rows: batch 1. TVOC: "
        "batch 2 (its policy is already the corrected one).", "",
        "### eCO2", "", *_status_table(eco2), "", "### TVOC", "", *_status_table(tvoc), "",
        "## Table 3. Cost per model in an edge-sized container (D = 7)", "",
        f"DALTON H1_41, 30-min horizon, D = 7 days; medians over test splits and {edge['config'].nunique()} Docker "
        "configurations (1-2 CPUs x 1-2 GB, which make no difference: every model is single-threaded). Train = wall "
        "time of one fit (Huber includes its internal CV). Peak Python memory = tracemalloc peak during the fit "
        "(misses native torch/LightGBM buffers; the process peak RSS, 642 MB, is dominated by loading the 1 Hz CSVs). "
        "The policy costs the sum of ridge and huber to train and one of them to predict.", "",
        *cost_table(edge), ""]
    return "\n".join(lines)


def build_paper(root: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    b1 = pd.read_csv(root / "summary_batch1_dalton_fraction.csv")
    b3 = pd.read_csv(root / "summary_batch3_dalton_fraction.csv")
    eco2 = pd.concat([b1[b1["model"].isin(["huber", "ridge", "lightgbm"])], b3[b3["model"] == "policy"]])
    tvoc = pd.read_csv(root / "summary_batch2_dalton_tvoc_fraction.csv")
    figs = {"fig_learning_curve": lambda: learning_curve_figure(eco2, tvoc),
            "fig_urbino_daytype": lambda: urbino_daytype_figure(pd.read_csv(root / "checks/urbino_daytype_calendar.csv")),
            "fig_alerts_tradeoff": lambda: alerts_tradeoff_figure(
                pd.read_csv(root / "summary_alerts_dalton_eco2_h30_overall.csv"))}
    written = []
    with plt.rc_context(PAPER_RC):
        for name, make in figs.items():
            fig = make()
            for ext in ("png", "svg"):
                fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
                written.append(out_dir / f"{name}.{ext}")
            plt.close(fig)
    (out_dir / "tables_paper.md").write_text(paper_tables(root), encoding="utf-8")
    return written + [out_dir / "tables_paper.md"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("fraction_csv", nargs="?", help="summarize.py *_fraction.csv")
    p.add_argument("--out", help="output path without extension")
    p.add_argument("--title")
    p.add_argument("--paper", action="store_true", help="final paper figures and tables from the committed summaries")
    p.add_argument("--results", default="results")
    p.add_argument("--out-dir", default="results/figures")
    a = p.parse_args()
    if a.paper:
        for f in build_paper(Path(a.results), Path(a.out_dir)):
            print("wrote", f)
        return
    if not (a.fraction_csv and a.out):
        p.error("fraction_csv and --out are required without --paper")
    frac = pd.read_csv(a.fraction_csv)
    fig = fraction_figure(frac, a.title)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {a.out}.png and {a.out}.pdf")


if __name__ == "__main__":
    main()
