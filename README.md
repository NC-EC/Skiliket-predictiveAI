# Skiliket-predictiveAI

Cold-start forecasting of indoor air quality with small local models.
SKILIKET project, Tec de Monterrey, team 9 (Modelos Predictivos).

**Research question:** how many days of data, and how much compute, does a small locally trained model need before it forecasts air-quality worsening (eCO2, TVOC, AQI) better than persistence on a brand-new sensor kit?

## Status
- [x] Canonical schema aligned with OGC SensorThings plus FAIR metadata (`src/coldstart/schema.py`)
- [x] SKILIKET adapter, verified against the real Zenodo file (`src/coldstart/adapters/skiliket.py`)
- [x] Evaluation core: rolling origin with purge, MASE vs persistence, baselines (`src/coldstart/evaluate.py`)
- [x] Cost measurement: wall/CPU time, memory, model size, inference latency (`src/coldstart/cost.py`)
- [x] Adapters for EDIAQI (22 devices), UCI Air Quality and QUANT, verified against the real files (`src/coldstart/adapters/`). QUANT and UCI have no eCO2/TVOC (covariates only)
- [x] Models: Ridge, random forest, LightGBM, tiny GRU (~1k params), all predicting change vs persistence (`src/coldstart/models.py`)
- [x] Alert evaluation: percentile threshold, recall, false alarms/day, lead time (`src/coldstart/alerts.py`)
- [x] Statistics: block bootstrap CI, cold-start criterion, Diebold-Mariano (`src/coldstart/stats.py`)
- [x] Learning-curve runner (`scripts/run_learning_curve.py`)
- [ ] Zero-shot pretrained model, edge simulation in Docker

## Quick start
```bash
pip install -e ".[dev,models]"
pytest
# put the SKILIKET xlsx in data/raw/skiliket_zenodo_17398551.xlsx, then:
python scripts/smoke_skiliket.py data/raw/skiliket_zenodo_17398551.xlsx   # pipeline check, NOT a result
python scripts/run_learning_curve.py --dataset skiliket --path data/raw/skiliket_zenodo_17398551.xlsx \
    --target eco2 --cadence 5 --horizon 15 --days 1 2 3 --seeds 3 --out results/smoke_skiliket.csv
```

## Data
Raw data is **not** redistributed. Download it with `scripts/download_datasets.sh` (EDIAQI and QUANT from Zenodo, CC BY 4.0; UCI Air Quality, research use) and the SKILIKET dataset from its Zenodo record.

## Conventions
Canonical channels: `eco2` (ppm), `tvoc` (ppb), `aqi` (sensor index 1–5), `temp_c` (°C), `rh_pct` (%).
Invalid readings are dropped (never imputed). Gaps longer than 5 min split a series into segments. The time zone stays `unknown` until it is confirmed.

## License
Code: MIT (see `LICENSE`).
