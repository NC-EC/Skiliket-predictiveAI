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
| EDIAQI indoor | Zenodo record 14763137 | CC BY 4.0 | 6 low-cost providers, TVOC (MOx), 5 s-5 min, 11-31 Oct 2024, labeled events JSON. Main dataset for alerts. Leave-one-device-out and leave-one-event-out. |
| QUANT | Zenodo record 10775692 | CC BY 4.0 | 49 commercial devices, CO2 but **no TVOC**, hourly CSV / minute NetCDF, 2019-2022. Leave-one-device-out. |
| UCI Air Quality | UCI id 360 | research use | Hourly, 9,358 rows 2004-2005, `-200` = missing (treat as invalid, drop). No eCO2/TVOC: map only what exists and document it. |
| SKILIKET | Zenodo (team) | CC BY 4.0 | Reference for channels/units; adapter done and verified. |

## CURRENT TASK (on the user's PC)
1. `pip install -e ".[dev,models]"` and run `pytest`.
2. Run `scripts/download_datasets.sh` (URLs are unverified, so fix them if needed). Read each `record.json`, download the data files it lists into `data/raw/<dataset>/`, and report file names, sizes and the first rows/columns of each.
3. Write `src/coldstart/adapters/ediaqi.py`, `quant.py` and `uci_air_quality.py` **against the real files**, following `skiliket.py`. Multi-device datasets: return a `device_id` column in addition to the canonical columns (or a dict of per-device frames; extend `schema.py` if needed and keep tests passing). Set `verified_against_real_file=True`.
4. Add a real-file test per adapter (skipped when the file is absent) that checks row counts, value ranges and units.
5. Smoke-run `run_learning_curve.py` on one EDIAQI device, then commit and push.
6. Afterwards: measure timing inside Docker with `--cpus 1|2 --memory 1g|2g` on this PC (Ryzen 5 5500 = timing reference).

## Open decisions (ask the user, don't assume)
Sensor model / time zone / device ID of SKILIKET kits; zero-shot pretrained model (Chronos vs TimesFM, check license and size); false-alarm budget; publication venue.
