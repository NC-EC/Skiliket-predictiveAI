#!/usr/bin/env bash
# Run LOCALLY (the cloud workspace cannot reach these hosts). Data is not redistributed in this repo.
# Verified 2026-10-28: record IDs and file names below match what the Zenodo / UCI hosts serve.
set -euo pipefail
mkdir -p data/raw/{ediaqi,uci,quant}

zenodo() {  # zenodo <record id> <dest dir> <file>...
  local rec=$1 dest=$2; shift 2
  curl -sSL --retry 3 "https://zenodo.org/api/records/$rec" -o "$dest/record.json"
  for f in "$@"; do
    curl -sSL --retry 3 "https://zenodo.org/api/records/$rec/files/$f/content" -o "$dest/$f"
  done
}

# EDIAQI indoor (Zenodo 14763137, CC BY 4.0): data.zip nests data/data/*.csv; flatten into data/raw/ediaqi/
zenodo 14763137 data/raw/ediaqi data.zip Measurement_metadata_description.xlsx
unzip -o -q -j data/raw/ediaqi/data.zip -d data/raw/ediaqi

# QUANT (Zenodo 10775692, CC BY 4.0): hourly long-format CSVs (~505 MB in total). No CO2 / TVOC in this dataset.
zenodo 10775692 data/raw/quant QUANT_SensorSystems_hourly.csv QUANT_Reference_hourly.csv QUANT_DuplicateRef_hourly.csv

# UCI Air Quality (research use only) - id 360
curl -sSL --retry 3 "https://archive.ics.uci.edu/static/public/360/air+quality.zip" -o data/raw/uci/air_quality.zip
unzip -o -q data/raw/uci/air_quality.zip -d data/raw/uci

ls -l data/raw/*
