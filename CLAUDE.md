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
Compare models with the median of `rel_mae` across splits, not the mean of MASE. **Cold-start criterion (user decision
2026-10-28): rel_mae < 1 with the upper 95% block-bootstrap CI bound < 1** (`stats.meets_cold_start`); MASE is secondary.

QUANT/UCI replaced by `adapters/dalton.py` and `adapters/urbino.py` with real-file tests; smoke runs
`results/smoke_urbino_olivetti.csv` (`--cadence 10 --horizon 30`) and `results/smoke_dalton_H1_41_h{15,30,60}.csv`
(`--cadence 5 --days 1 3 7 14`; ~24 days of grid, so 25/23/19/12 splits; ~8 min per horizon on the Ryzen 5 5500).
Median rel_mae (smoke, not results):
- Urbino olivetti, h30, D=1/3/7: LightGBM 1.00/1.00/0.99 (wins 50/45/60 % of splits), Ridge 1.01/1.00/1.00.
- DALTON H1_41, D=1/3/7/14: LightGBM h15 1.05/1.00/0.99/1.00, h30 1.05/1.00/0.99/1.00, h60 1.04/1.02/0.97/1.00
  (wins 72 % of splits at h60 D=7). Ridge gets worse with more days (h15 1.00 -> 1.19; cause not yet checked).
  RF 1.1-1.6 and GRU 1.05-1.4 lose everywhere; moving average ~1.0-1.1.
Only LightGBM approaches persistence, and only from D=3; check it on more devices before reading anything into it.

## CURRENT TASK (on the user's PC), in this order
Model fixes before any real experiment (add a unit test for each, `pytest`, commit + push after each):
1. **Validation gap**: `_val_split` and `TimeSeriesSplit` use a fixed gap of 12 windows. With 6 lags and a 60-min horizon at
   5-min cadence a window spans 18 steps, so fit/validation windows overlap and model selection is optimistic. Make the gap
   `n_lags + horizon_steps`, passed from the runner to every model.
2. **LightGBM refit**: after early stopping on the training tail, refit on ALL training windows with `best_iteration_`
   trees. Today the most recent 10 % of history (the most relevant in cold start) is never used for the final model.
3. **Linear model with a robust loss**: add `huber` (sklearn `HuberRegressor`, epsilon chosen by the same time-ordered CV on
   MAE) next to `ridge`. Ridge gets WORSE with more days on DALTON H1_41 (rel_mae 1.00 -> 1.19 at 15 min); the change
   y - y0 is heavy-tailed, so an L2 fit chases the spikes of older days. Keep ridge in the comparison.
4. **Calendar features**: add hour-of-day as sin/cos and a weekend flag to X (option `--calendar`, default on; computed from
   the origin time stamp, so no leakage). Indoor CO2 follows occupancy schedules; without time of day a model cannot learn
   the daily pattern that more days of history should reveal. This is the main mechanism the learning curve tests.
5. **Sanity check**: moving_average median rel_mae on H1_41 at 30 min is exactly 1.049 in all four day columns while the
   test-split sets differ (25/23/19/10 splits). Confirm the per-split values really differ; if not, find why.
6. Re-run DALTON H1_41 (days 1 3 7 14, horizons 15/30/60) and show the same median rel_mae table plus % of splits won,
   before vs after these fixes. Then run 3 more DALTON devices from different site types (a home, a lab, a classroom).
7. Random forest and GRU lose everywhere: keep them in the paper as reported results, no more tuning time.
8. Afterwards: timing inside Docker with `--cpus 1|2 --memory 1g|2g` (Ryzen 5 5500 = timing reference).

## Open decisions (ask the user, don't assume)
Sensor model / time zone / device ID of SKILIKET kits; zero-shot pretrained model (Chronos vs TimesFM, check license and size); false-alarm budget; publication venue.
