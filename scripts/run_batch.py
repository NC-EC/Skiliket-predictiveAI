"""Run a batch of learning-curve jobs with N parallel runner processes; resumable (finished outputs are skipped).

Batch 1: DALTON eco2 (every device, h15/30/60).
Batch 3: DALTON eco2 again with ridge, huber and the corrected policy only (results/runs_batch3/).
Batch 2: DALTON tvoc, EDIAQI (providers with TVOC in ppb; eco2 and tvoc) and Urbino (3 rooms, eco2).
Cadence per device from its native time step (native_cadence): 5 min when the grid would not split on its steps,
else 10 min, and a 10-min device runs only h30 and h60. Urbino steps 5 min but 8-9 % of its steps are 10 min, which at
a 5-min grid cut every segment to about 1 h (batch 1 ran it that way; its h15 output was discarded): it runs at 10 min.
DALTON is a 1 Hz logger, so its devices are 5 min without loading the raw files here.
Jobs run horizon by horizon, longest devices first, so a device's grid cache (data/cache/) is built by its first job
and reused by the other horizons. Cost columns from these runs are not timing (parallel > 1).

  python scripts/run_batch.py --batch 1 --workers 5            # results/runs/*.csv, logs in results/runs/logs/
  python scripts/run_batch.py --batch 1 --dry-run              # list the jobs
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parents[1]
MODELS = ["ridge", "huber", "random_forest", "lightgbm", "gru", "policy"]
# batch 3 re-runs DALTON eco2 with the corrected policy (ridge/huber candidates, cv_rel_mae column); RF, LightGBM and GRU
# keep their batch-1 rows. Its own folder, so the resumable skip does not mistake batch-1 outputs for finished jobs.
BATCH_MODELS = {1: MODELS, 2: MODELS, 3: ["ridge", "huber", "policy"]}
BATCH_OUT = {1: "results/runs", 2: "results/runs", 3: "results/runs_batch3"}
HORIZONS = {5: [15, 30, 60], 10: [30, 60]}  # cadence (min) -> horizons (a multiple of the cadence)
CADENCE_SHARE = 0.99  # share of time steps that must fit the grid's segment limit (1.5 x cadence, evaluate.to_grid)


def native_cadence(ts) -> int | None:
    """5 or 10 (minutes): the finest grid whose segment limit (1.5 x cadence) holds >= CADENCE_SHARE of the device's
    time steps; None when neither does."""
    import pandas as pd

    steps = pd.Series(pd.to_datetime(ts)).sort_values().diff().dt.total_seconds().div(60).dropna()
    for c in (5, 10):
        if len(steps) and (steps <= 1.5 * c).mean() >= CADENCE_SHARE:
            return c
    return None


def _cadence(adapter, path, device):
    c = native_cadence(adapter.load(path, device)[0]["ts"])
    if c is None:
        print(f"skipped {adapter.__name__.rsplit('.', 1)[-1]} {device}: native step coarser than 10 min")
    return c


def _dalton(target):
    from coldstart.adapters import dalton

    path = ROOT / "data/raw/dalton"
    # day files per device: a proxy for length, used only to start the long jobs first
    size = {d: len(list((path / "Processed" / d.split("_")[0]).glob(f"*/{d.split('_', 1)[1]}_*.csv")))
            for d in dalton.devices(path)}
    return [("dalton", "data/raw/dalton", d, target, 5, ["1", "2", "3", "5", "7", "14"])
            for d in sorted(size, key=size.get, reverse=True)]


def _urbino():
    from coldstart.adapters import urbino

    path = ROOT / "data/raw/urbino"
    cad = {d: _cadence(urbino, path, d) for d in ["olivetti", "turing", "von_neumann"]}
    return [("urbino", "data/raw/urbino", d, "eco2", c, ["1", "3", "7", "14", "30", "60"]) for d, c in cad.items() if c]


def _ediaqi():
    from coldstart.adapters import ediaqi

    path = ROOT / "data/raw/ediaqi"
    devs = [d for d in ediaqi.devices(path) if d.startswith(("provider1", "provider3", "provider4", "provider6"))]
    cad = {d: _cadence(ediaqi, path, d) for d in devs}
    return [("ediaqi", "data/raw/ediaqi", d, t, cad[d], ["1", "2", "3", "5", "7"])
            for t in ("eco2", "tvoc") for d in devs if cad[d]]


def jobs(batch: int):
    specs = _dalton("eco2") if batch in (1, 3) else _urbino() + _dalton("tvoc") + _ediaqi()
    js = [(*s, h) for h in HORIZONS[5] for s in specs if h in HORIZONS[s[4]]]
    # Urbino first: its rooms span months (the longest jobs, ~6-10 h), so they must not be the tail of the batch
    return sorted(js, key=lambda j: j[0] != "urbino")


def run(job, workers: int, out_dir: Path, models=MODELS) -> tuple[str, int, float]:
    dataset, path, device, target, cadence, days, h = job
    tag = f"{dataset}_{device}_{target}_h{h}"
    out = out_dir / f"{tag}.csv"
    if out.exists():
        return tag, 0, 0.0
    cmd = [sys.executable, "-W", "ignore", str(ROOT / "scripts/run_learning_curve.py"), "--dataset", dataset,
           "--path", path, "--device", device, "--target", target, "--cadence", str(cadence), "--horizon", str(h),
           "--days", *days, "--seeds", "3", "--models", *models, "--parallel", str(workers), "--out", str(out)]
    t = time.time()
    with open(out_dir / "logs" / f"{tag}.log", "w") as log:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    return tag, rc, time.time() - t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--out-dir", help="default: results/runs (batches 1-2), results/runs_batch3 (batch 3)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    js = jobs(a.batch)
    if a.dry_run:
        print("\n".join(" ".join(map(str, (*j[:4], f"c{j[4]}", f"h{j[6]}"))) for j in js), f"\n{len(js)} jobs")
        return
    out_dir = ROOT / (a.out_dir or BATCH_OUT[a.batch])
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as pool:
        futures = [pool.submit(run, j, a.workers, out_dir, BATCH_MODELS[a.batch]) for j in js]
        for i, f in enumerate(as_completed(futures), 1):
            tag, rc, dt = f.result()
            status = "ok" if rc == 0 else f"exit {rc} (see logs/{tag}.log)"
            print(f"[{i}/{len(js)}] {time.strftime('%H:%M')} {tag} {dt / 60:.1f} min {status}", flush=True)
    print(f"done in {(time.time() - t0) / 3600:.1f} h")


if __name__ == "__main__":
    main()
