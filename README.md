# Skiliket-predictiveAI

Cold-start forecasting of indoor air quality with small local models.
SKILIKET project, Tec de Monterrey, team 9 (Modelos Predictivos).

**Research question:** how many days of data, and how much compute, does a small locally trained model need before it forecasts air-quality worsening (eCO2, TVOC, AQI) better than persistence on a brand-new sensor kit?

## Status
- [x] Canonical schema aligned with OGC SensorThings plus FAIR metadata (`src/coldstart/schema.py`)
- [x] SKILIKET adapter, verified against the real Zenodo file (`src/coldstart/adapters/skiliket.py`)
- [x] Evaluation core: rolling origin with purge, MASE vs persistence, baselines (`src/coldstart/evaluate.py`)
- [x] Cost measurement: wall/CPU time, memory, model size, inference latency (`src/coldstart/cost.py`)
- [ ] Adapters for EDIAQI, UCI Air Quality and QUANT (need the real files)
- [ ] Models: Ridge, trees, small GRU/CNN, zero-shot pretrained model
- [ ] Alert evaluation, edge simulation in Docker

## Quick start
```bash
pip install -e ".[dev]"
pytest
# put the SKILIKET xlsx in data/raw/skiliket_zenodo_17398551.xlsx, then:
python scripts/smoke_skiliket.py data/raw/skiliket_zenodo_17398551.xlsx   # pipeline check, NOT a result
```

## Data
Raw data is **not** redistributed. Download it with `scripts/download_datasets.sh` (EDIAQI and QUANT from Zenodo, CC BY 4.0; UCI Air Quality, research use) and the SKILIKET dataset from its Zenodo record.

## Conventions
Canonical channels: `eco2` (ppm), `tvoc` (ppb), `aqi` (sensor index 1–5), `temp_c` (°C), `rh_pct` (%).
Invalid readings are dropped (never imputed). Gaps longer than 5 min split a series into segments. The time zone stays `unknown` until it is confirmed.

## License
Code: MIT (see `LICENSE`).
