"""Run a batch of learning-curve jobs with N parallel runner processes; resumable (finished outputs are skipped).

Batch 1: DALTON eco2 (every device, h15/30/60) and Urbino (3 rooms, eco2, h15/30/60).
Batch 2: DALTON tvoc and EDIAQI (providers with TVOC in ppb), eco2 and tvoc.
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
HORIZONS = [15, 30, 60]


def _dalton(target):
    from coldstart.adapters import dalton

    path = ROOT / "data/raw/dalton"
    # day files per device: a proxy for length, used only to start the long jobs first
    size = {d: len(list((path / "Processed" / d.split("_")[0]).glob(f"*/{d.split('_', 1)[1]}_*.csv")))
            for d in dalton.devices(path)}
    return [("dalton", "data/raw/dalton", d, target, ["1", "2", "3", "5", "7", "14"])
            for d in sorted(size, key=size.get, reverse=True)]


def _urbino():
    return [("urbino", "data/raw/urbino", d, "eco2", ["1", "3", "7", "14", "30", "60"])
            for d in ["olivetti", "turing", "von_neumann"]]


def _ediaqi():
    from coldstart.adapters import ediaqi

    devs = [d for d in ediaqi.devices(ROOT / "data/raw/ediaqi") if d.startswith(("provider1", "provider3",
                                                                                   "provider4", "provider6"))]
    return [("ediaqi", "data/raw/ediaqi", d, t, ["1", "2", "3", "5", "7"]) for t in ("eco2", "tvoc") for d in devs]


def jobs(batch: int):
    specs = _dalton("eco2") + _urbino() if batch == 1 else _dalton("tvoc") + _ediaqi()
    return [(*s, h) for h in HORIZONS for s in specs]


def run(job, workers: int, out_dir: Path) -> tuple[str, int, float]:
    dataset, path, device, target, days, h = job
    tag = f"{dataset}_{device}_{target}_h{h}"
    out = out_dir / f"{tag}.csv"
    if out.exists():
        return tag, 0, 0.0
    cmd = [sys.executable, "-W", "ignore", str(ROOT / "scripts/run_learning_curve.py"), "--dataset", dataset,
           "--path", path, "--device", device, "--target", target, "--cadence", "5", "--horizon", str(h),
           "--days", *days, "--seeds", "3", "--models", *MODELS, "--parallel", str(workers), "--out", str(out)]
    t = time.time()
    with open(out_dir / "logs" / f"{tag}.log", "w") as log:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    return tag, rc, time.time() - t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, choices=[1, 2], required=True)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--out-dir", default="results/runs")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    js = jobs(a.batch)
    if a.dry_run:
        print("\n".join(" ".join(map(str, j[:4] + (f"h{j[5]}",))) for j in js), f"\n{len(js)} jobs")
        return
    out_dir = ROOT / a.out_dir
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as pool:
        futures = [pool.submit(run, j, a.workers, out_dir) for j in js]
        for i, f in enumerate(as_completed(futures), 1):
            tag, rc, dt = f.result()
            status = "ok" if rc == 0 else f"exit {rc} (see logs/{tag}.log)"
            print(f"[{i}/{len(js)}] {time.strftime('%H:%M')} {tag} {dt / 60:.1f} min {status}", flush=True)
    print(f"done in {(time.time() - t0) / 3600:.1f} h")


if __name__ == "__main__":
    main()
