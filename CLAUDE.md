# Context for Claude Code

## Project
SKILIKET (Tec de Monterrey), team 9 "Modelos Predictivos". Research question for an English experimental paper:
**how many days of data and how much compute does a small, locally trained model need to forecast indoor air-quality
worsening (eCO2, TVOC, AQI; covariates temperature and RH) better than persistence on a brand-new sensor kit (cold start)?**
The user speaks Spanish; reply in Spanish. Code, comments and the paper are in English.

## Hard rules
- **Real data only.** Never invent or simulate data for results. Synthetic data is allowed only inside unit tests and must be labeled.
- **Do not commit raw data** (`data/raw/` is gitignored). Datasets are downloaded by script.
- **Never impute.** Invalid readings are dropped; gaps > 5 min split a series into `segment`s; windows never cross segments.
- **No leakage.** Splits use `evaluate.rolling_origin` with a purge of horizon + lag window. Alert thresholds and any tuning come from training data only.
- Smoke runs are not results; name their outputs `results/smoke_*.csv`.
- Flag anything you could not verify (units, time zone, sensor model) in `DatasetMeta.unit_assumptions` / `notes`, and keep `timezone="unknown"` unless the source states it.
- Run `pytest` before every commit. Commit in small steps and push to `main`.

## Layout
- `src/coldstart/schema.py`: canonical channels (`eco2` ppm, `tvoc` ppb, `aqi` 1-5, `temp_c` degC, `rh_pct` %), `DatasetMeta`, `validate()`.
- `src/coldstart/adapters/`: one module per dataset exposing `load(path) -> (df, DatasetMeta, report)`. `skiliket.py` is the reference implementation; reuse `base.finalize()`.
- `src/coldstart/evaluate.py`: gridding, windows, rolling origin, MASE/MAE/RMSE, baselines.
- `src/coldstart/models.py`: Ridge, RandomForest, LightGBM, tiny GRU, all predicting the change vs persistence.
- `src/coldstart/alerts.py`, `stats.py`, `cost.py`: alerts, bootstrap CI / cold-start criterion / Diebold-Mariano, compute cost.
- `scripts/run_learning_curve.py`: main experiment runner (`--dataset <adapter module name>`).
- `config.yaml`: experiment factors.

## Datasets
| Dataset | Source | License | Notes |
|---|---|---|---|
| EDIAQI indoor | Zenodo record 14763137 | CC BY 4.0 | One house in Zagreb, 11-31 Oct 2024. 22 devices (providers 1-6, some multi-unit, plus a reference instrument), 5 s-10 min. Local Zagreb wall clock; the DST hour 2024-10-27 02:00-03:00 is dropped. TVOC in ppb only for providers 1, 3, 4 (ppm x1000), 6. **No labeled-events JSON** in this record (only People/Door/Motion logs). Main dataset for alerts. Leave-one-device-out. |
| DALTON | github.com/prasenjit52282/dalton-dataset (CSVs in Git LFS, ~13 GB) | AGPL-3.0 (authors: free for non-commercial research); cite the paper (arXiv 2407.14501) | India, 30 indoor sites, 68 distinct devices (F2_24 duplicates F1_23 reading for reading), 2022-2024, 1 Hz, ~27,800 usable hours. CO2 ppm, VoC ppb, T degC, H % per README. Time zone not stated (ESP32 clock): `unknown`. **The authors' `Processed/` files are imputed** (`library/preprocess.py`): gaps <= 15 min nearest-filled and marked `Valid=1`, all channels smoothed with a trailing 60 s mean, longer gaps set to 0. `adapters/dalton.py` drops Valid/Valid_CO2 = 0, zeros and the 60 s ramps after them, all-channel freezes > 60 s, deep dips (CO2 < 250, T < 5, RH < 5), CO2 = 8191, and 1970/2028 clock errors. Short fills and shallow dips stay undetectable (~1-2 % of rows): **report this as a limitation in the paper** (user chose this over the raw files, which contain participants' phone numbers). 3,957 activity annotations via `dalton.load_events(path, site)`. Main dataset for leave-one-site-out. |
| Urbino classrooms | Zenodo record 17487080 | CC BY 4.0 | Italy, 3 university classrooms (olivetti, turing, von_neumann), calendar 2024. Files step every **5 min** (record says 10), UTC time stamps ('Z'). CO2 (5000 ppm = sensor ceiling, dropped), T, RH; **no TVOC**. Units not stated anywhere: ppm/degC/% inferred. Long learning curves (weeks to months) and seasonality; horizons 30/60 min, `--cadence 10`. |
| SKILIKET | Zenodo (team) | CC BY 4.0 | Reference for channels/units; adapter done and verified. |

QUANT and UCI Air Quality were **dropped** (2026-10-28, user decision): both are outdoor and have no CO2/TVOC.

## Environment (user's PC, Windows)
The repo lives at `C:\Users\netop\dev\Skiliket-predictiveAI`, outside OneDrive (DALTON is ~13 GB and must not sync).
The only system Python is MSYS2's, which cannot install torch/lightgbm wheels. Use the uv-managed venv:
`uv venv --python 3.12 .venv`, then `uv pip install --python .venv\Scripts\python.exe -e ".[dev,models]"` (CPU torch from
`https://download.pytorch.org/whl/cpu`). Run everything with `.venv\Scripts\python.exe`.

## Done (2026-10-28)
Install + pytest; `download_datasets.sh` verified and now fetches/unzips everything; adapters `ediaqi.py` (per-device,
`load(path, device)` / `load_all(path)`), `quant.py` (same API), `uci_air_quality.py`, with real-file tests; `--device` in the
runner; smoke run on EDIAQI `provider1` (`results/smoke_ediaqi_provider1.csv`). Coarse devices use a max gap of
max(5 min, 1.5 x native step), mirroring `evaluate.to_grid`.

MASE diagnosis (EDIAQI provider1): align = 0, no code bug. RF/LightGBM overfit (in 0.13-0.75, out > 1) and test days
often differ from training days (regime shift). Ridge's in-sample MAE ratio > 1 on 26/54 splits is NOT a bug: its
in-sample RMSE ratio is always < 1 (max 0.957); the change y - y0 is heavy-tailed (std/MAE 2.8 vs 1.25 Gaussian), so an
L2 fit can lose to zero change on MAE. Regularization applied in `models.py` (time-ordered CV alpha for Ridge on MAE,
RF min_samples_leaf=20, LightGBM 7 leaves + early stopping on the training tail). Median rel_mae on provider1:
LightGBM ~2.0 -> 1.02-1.07, Ridge 1.46-2.04 -> 1.02-1.19, RF 2.2 -> 1.6-1.7. Nothing beats persistence yet at 15 min.
Compare models with the median of `rel_mae` across splits, not the mean of MASE. **Cold-start criterion (revised 2026-10-28 evening): GEOMETRIC mean of rel_mae < 1 with the upper 95% block-bootstrap
CI bound < 1**, after dropping test splits shorter than half a full test day (`summarize.py --min-test-frac 0.5`). Reason: one
10-window split on H1_41 gave rel_mae 33 and wrecked the arithmetic mean of every model. Also reported: median,
arithmetic mean (secondary) and pooled rel_mae. Implemented in `stats.geomean_ci` / `stats.meets_cold_start` and `summarize.py`;
no re-run needed, it only changes how runner CSVs are summarized.

QUANT/UCI replaced by `adapters/dalton.py` and `adapters/urbino.py` with real-file tests; smoke runs
`results/smoke_urbino_olivetti.csv` (`--cadence 10 --horizon 30`) and `results/smoke_dalton_H1_41_h{15,30,60}.csv`
(`--cadence 5 --days 1 3 7 14`; ~24 days of grid, so 25/23/19/12 splits; ~8 min per horizon on the Ryzen 5 5500).
Median rel_mae (smoke, not results):
- Urbino olivetti, h30, D=1/3/7: LightGBM 1.00/1.00/0.99 (wins 50/45/60 % of splits), Ridge 1.01/1.00/1.00.
- DALTON H1_41, D=1/3/7/14: LightGBM h15 1.05/1.00/0.99/1.00, h30 1.05/1.00/0.99/1.00, h60 1.04/1.02/0.97/1.00
  (wins 72 % of splits at h60 D=7). Ridge gets worse with more days (h15 1.00 -> 1.19; cause not yet checked).
  RF 1.1-1.6 and GRU 1.05-1.4 lose everywhere; moving average ~1.0-1.1.
Only LightGBM approaches persistence, and only from D=3; check it on more devices before reading anything into it.

Model fixes 1-5 done (2026-10-28): validation gap = n_lags + horizon_steps passed to every model; LightGBM refits on all
windows with the early-stopped tree count; `huber` model (epsilon by time-ordered CV on MAE); calendar features
(`evaluate.calendar_features`, `--calendar` on by default; learned models only, the GRU gets them as static channels);
moving-average check: no bug. Baselines do not train, so a test day gives the same value for every D, and the test days
for a larger D are the last ones of D=1 (the 1.049 median day sits in the middle of all four sets by chance).

Smoke re-runs after the fixes, DALTON, `--cadence 5 --days 1 3 7 14`, horizons 15/30/60
(`results/smoke_dalton_<device>_h<h>.csv`; H1_41 before = `*_before.csv`). The 4 devices ran in parallel, so their cost
columns are not timing. Median rel_mae, D=1/3/7/14 (smoke, not results):
- H1_41 (home, 24 days), before -> after: LightGBM about the same (h60 1.04/1.02/0.97/1.00 -> 1.02/0.98/0.98/0.99);
  ridge still worsens with days (h15 1.00 -> 1.16); huber h60 1.06/0.91/0.87/1.00; GRU worse than before at every cell
  (1.3-1.9; gap and/or calendar channels, not investigated per item 7); RF slightly better.
- H8_81 (home, 24 days): huber h30 1.16/0.88/0.94/0.78 (D=14 wins 9/9 splits); LightGBM 1.00/1.14/1.00/0.93.
- R1_51 (lab, 35 days): clearest learning curve. huber h15 1.03/0.74/0.73/0.70, ridge 1.01/0.97/0.89/0.73, LightGBM
  1.00/0.98/0.95/0.91; RF and GRU also drop below 1 from D=7.
- C1_25 (classroom, 14 usable days): no D=14 split; huber h15 1.56/0.95/0.67 (7 splits at D=7).
Pattern: with D=1 every learned model is >= persistence (huber worst, 1.15-2.2); from D=3 the linear models with calendar
features beat persistence on 3 of 4 devices. Not yet checked against the cold-start criterion (CI bound).

## Done 2026-10-28 (afternoon, user's PC)
- `--common-test` is the default (`--no-common-test` for smoke). The runner drops train lengths a device cannot hold
  (D + test_days > span), so short devices keep their feasible D instead of losing every split. CSVs now carry
  `device` and `policy_choice`.
- Huber alpha by CV (`results/checks/`, H1_41 + R1_51, h15, D=1/3/7): helps H1_41 (median rel_mae D=3 1.09 -> 0.99)
  but hurts R1_51 at D=3 (0.756 -> 0.935): **not kept**, alpha stays 1e-4. Note: fixed-alpha huber has outlier splits
  on H1_41 (mean rel_mae 2.8-3.0 vs median 1.0-1.15).
- `scripts/summarize.py` (cells + D*) with tests; `policy` model (ridge/huber/lightgbm on the training tail
  (20 %, same gap), else persistence; one seed).
- Edge timing, Docker (`results/edge/`, DALTON H1_41, h30, D=1/3/7, 19 splits, 5 models, libraries pinned to the host
  versions): 10.6-10.9 min per configuration; **CPU 1 vs 2 and 1 vs 2 GB make no difference** (all models single-thread,
  n_jobs=1). Median train s at D=7: huber 5.3 (its CV, 26 lbfgs fits), RF 1.75, GRU 0.43, ridge 0.30, LightGBM 0.03.
  Inference per window: ridge/huber 0.09 ms, GRU 0.22, LightGBM 0.40, RF 6.6 ms. Model size: ridge 1.3 kB, huber 2.2,
  GRU 6.0, LightGBM 9.3, RF 240 kB. Process peak RSS 642 MB in every configuration, dominated by loading 1 Hz DALTON
  CSVs, not by the models; `peak_py_mb` (tracemalloc, 0.3-2 MB) misses native torch/LightGBM buffers. A per-model
  memory number needs a fresh process per fit on pre-gridded data (not done).

## Done 2026-10-28 (evening)
- Change clip [p1, p99] (`_ChangeModel.clip`): implemented and tested, **off by default**: on H1_41 h15 it left huber's
  mean rel_mae unchanged (3.00 vs 2.98). The mean comes from one test split with 10 windows (persistence MAE 1.35,
  huber rel_mae 33 at every D). Open: a minimum test size per split, applied in summarize (n_test is in every CSV).
- Huber speed-up not adopted: max_iter=100 is 4x faster but changes rel_mae (R1_51 D=1 median 1.04 -> 1.15);
  tol=1e-3 changes nothing (lbfgs stops on max_iter).
- `policy` picks by the candidates' own validation scores (`cv_rel_mae_`: ridge/huber CV folds, LightGBM early-stopping
  tail) and reuses the runner's fitted models: no extra training. Its cost row = sum of the candidates' times.
- Grid cache in `data/cache/` (`coldstart.cache`, keyed by a hash of adapter + base + evaluate code).
- **Cost for the paper: `peak_py_mb` of the fit** (summarize reports medians of train_s, peak_py_mb, infer_ms,
  model_bytes), not the process RSS.
- Small check (C1_25 h15, all models): runs end to end; D* = NaN for all (nothing meets the criterion there); on the
  R1_51 check CSV huber D* = 3 as expected. Policy loses at D <= 3 on C1_25 (picks LightGBM on a CV score < 1, then
  loses on test): the validation score is optimistic on very short histories.

## Done 2026-10-28 (night) / 2026-10-29
- Cloud commits merged: policy candidates = ridge + huber only (LightGBM's score is optimistic: validated on its own
  early-stopping tail); the runner writes `cv_rel_mae`; `summarize.py` recomputes the policy from it, only for the
  files that carry the column (older CSVs keep the runner's policy rows). Stable D* statuses: `reached`,
  `unconfirmed` (D* = largest D tested), `insufficient` (one D tested), `never` (tested at the design max, no stable D),
  `untestable` (too short for the design max). `scripts/figures.py`: share of devices meeting the criterion vs D.
- DALTON H2_16 / R4_21 (0 days): 86-92 % of their rows are Valid=0 or zero-filled by the authors (4.6 h and 7 h
  survive) and the all-channel freeze rule removes ~70 % of the rest (junk such as CO2 = 15). Adapter unchanged.
  11 DALTON devices hold < 2 days and are excluded by the runner: A5_121 0.0, H2_16 0.0, R4_21 0.0, H10_103 0.9,
  H10_104 0.8, H11_106 1.4, H11_107 1.4, H13_95 1.8, H13_96 1.8, A6_122 2.0, A7_113 2.0 days. R4_22 / R4_23 have 2.2
  days at h15 and none at h30/h60.
- **Batch 1 done** (2026-10-28 16:26 -> 10-29 05:47, 13.3 h, 5 workers): DALTON eco2, 57 devices x h15/30/60, 167
  CSVs in `results/runs/` (gitignored). Summary committed: `results/summary_batch1_dalton_{cells,dstar,fraction}.csv`
  (12,040 rows dropped as short test splits). Batch-1 policy rows are the OLD policy (ridge/huber/lightgbm), except
  R4_24 h60 (re-run with the new code); batch 3 replaces them.
  Devices meeting the criterion at D = 1/2/3/5/7/14 (%): huber h15 5/17/24/34/42/56, h30 4/13/20/26/31/47,
  h60 0/8/10/20/28/45; ridge h15 2/6/18/30/24/38; LightGBM h60 5/15/16/42/40/42. Devices tested per D (h15):
  57/52/51/50/45/34. Stable D* reached (median D*): huber 21 (3) / 14 (3) / 12 (5) at h15/30/60, ridge 14/15/17 (5).
- **Urbino in batch 1 was wrong**: run at `--cadence 5`, but 8-9 % of its steps are 10 min, so the 5-min grid cut
  every segment to ~1 h (7709 segments, none >= 3 h; at 10 min: 128 segments, median 17 h). h30/h60 had ~1 window and
  h15 used only 1-h fragments: h15 output moved to `results/runs/discarded_urbino_c5/` and not used.
- `run_batch.py` picks the cadence per device (`native_cadence`): 5 min if >= 99 % of the raw time steps are <= 7.5 min
  (the grid's segment limit), else 10 min (then only h30/h60). Urbino -> 10; all EDIAQI providers 1/3/4/6 -> 5
  (99.3-100 %); DALTON is 1 Hz, fixed at 5 without loading.
- LightGBM with < 20 validation (or fit) windows trains a fixed 100 trees, no early stopping, `cv_rel_mae_` NaN (a
  1-window tail crashed DALTON R4_24 h60). The runner logs an error in one D and keeps the other D (the failed D writes
  no rows; "FAILED train lengths" at the end of the job log). Grid cache writes are atomic (tmp + rename).

## CURRENT TASK
**Batch 2 running** (`scripts/run_batch.py --batch 2 --workers 5`, log `results/runs/batch2.log`; resumable): Urbino
eco2 at 10 min x h30/h60 (D=1/3/7/14/30/60; first, longest jobs), DALTON tvoc (68 devices x h15/30/60), EDIAQI
providers 1/3/4/6 (16 devices x eco2/tvoc x h15/30/60, D=1/2/3/5/7). 306 jobs.
Then batch 3 (give the user an estimate first): DALTON eco2 h15/30/60, `--models ridge huber policy`, so the policy is
the new one with `cv_rel_mae`. Random forest and GRU stay as reported results, no more tuning time.

**Alerts (code done 2026-10-29, cloud; not run yet):** `scripts/run_alerts.py` (same splits as the learning curve;
threshold p90/p95 of the D training days; decision offset chosen on a 30 % training tail at FA/day <= budget 0.5/1/2,
then the model is refit on all D days and tested with that offset; also `raw` = offset 0; `alarm_frac` because merged
alarms make an all-day alarm count as one false alarm; `--events` flags test events with a DALTON annotation within
[onset - 60, onset + 15] min), `scripts/summarize_alerts.py`, `scripts/check_annotation_clock.py` (scans annotation
clock shifts of -12..+12 h before trusting the annotations; run it first, it only reads the grid cache).
Clock check done (`results/checks/annotation_clock.csv`, 38 devices with annotations): no clock offset (median share
0.44 at 0, peak 0.46 at -30 min, no peak near +-330 min), so `--ann-shift-min 0`. But the link is weak: chance level
(shifts >= 4 h) is 0.32, so only +0.10 above chance; 26/38 devices above their own chance level. Annotations are dense
(median 218 per device), so they are a descriptive check of alerts, not ground truth for events.
Plan: alerts after batch 2: DALTON eco2 h30, D=3/7/14, persistence/ridge/
huber/lightgbm/policy, `--events --ann-shift-min <peak if clearly != 0>`.

Notes for the paper (do not act on them now): DALTON's clock time zone is unknown, so hour-of-day features are in each
device's own clock (fine for per-device models, not comparable across sites); calendar features are what turns the
learning curve upward from D = 3 on the 4 smoke devices.

## Open decisions (ask the user, don't assume)
Sensor model / time zone / device ID of SKILIKET kits; zero-shot pretrained model (Chronos vs TimesFM, check license and size); false-alarm budget; publication venue.
