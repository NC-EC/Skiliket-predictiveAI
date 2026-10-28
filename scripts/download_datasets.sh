#!/usr/bin/env bash
# Run LOCALLY (the cloud workspace cannot reach these hosts). Data is not redistributed in this repo.
# EDIAQI part verified 2026-10-28 on the user's PC. Urbino and DALTON parts added from the cloud session, not yet run.
set -euo pipefail
mkdir -p data/raw/{ediaqi,urbino}

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

# Urbino classrooms (Zenodo 17487080, CC BY 4.0). File names taken from the record page (unverified): check record.json.
zenodo 17487080 data/raw/urbino olivetti.csv turing.csv von_neumann.csv

# DALTON (GitHub, AGPL-3.0; non-commercial research). CSVs are in Git LFS: requires `git lfs install` once.
if [ ! -d data/raw/dalton/.git ]; then
  git clone --depth 1 https://github.com/prasenjit52282/dalton-dataset data/raw/dalton
fi
git -C data/raw/dalton lfs pull

ls -l data/raw/*
