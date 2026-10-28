#!/usr/bin/env bash
# Run LOCALLY (the cloud workspace cannot reach these hosts). Data is not redistributed in this repo.
# UNVERIFIED: record IDs/URLs come from the dataset pages found during the proposal; confirm file names after download.
set -euo pipefail
mkdir -p data/raw/{ediaqi,uci,quant}
# EDIAQI indoor (Zenodo 14763137, CC BY 4.0)
curl -L "https://zenodo.org/api/records/14763137" -o data/raw/ediaqi/record.json
# list files then download: python -c "import json;[print(f['key'],f['links']['self']) for f in json.load(open('data/raw/ediaqi/record.json'))['files']]"
# QUANT (Zenodo 10775692, CC BY 4.0)
curl -L "https://zenodo.org/api/records/10775692" -o data/raw/quant/record.json
# UCI Air Quality (research use only) - id 360
curl -L "https://archive.ics.uci.edu/static/public/360/air+quality.zip" -o data/raw/uci/air_quality.zip
echo "Now inspect record.json files, download the data files they list, and run the adapters (to be written against the real files)."
