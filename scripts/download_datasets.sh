#!/usr/bin/env bash
# Run LOCALLY (the cloud workspace cannot reach these hosts). Data is not redistributed in this repo.
# Verified 2026-10-28 on the user's PC (EDIAQI, Urbino, DALTON). DALTON needs `git lfs` and ~13 GB (keep data/raw outside OneDrive).
set -euo pipefail
mkdir -p data/raw/{ediaqi,urbino}

zenodo() {  # zenodo <record id> <dest dir> <file>...
  local rec=$1 dest=$2; shift 2
  curl -fsSL --retry 3 "https://zenodo.org/api/records/$rec" -o "$dest/record.json"
  for f in "$@"; do
    curl -fsSL --retry 3 "https://zenodo.org/api/records/$rec/files/$f/content" -o "$dest/$f"
  done
}

# EDIAQI indoor (Zenodo 14763137, CC BY 4.0): data.zip nests data/data/*.csv; flatten into data/raw/ediaqi/
zenodo 14763137 data/raw/ediaqi data.zip Measurement_metadata_description.xlsx
unzip -o -q -j data/raw/ediaqi/data.zip -d data/raw/ediaqi

# Urbino classrooms (Zenodo 17487080, CC BY 4.0). File names checked against record.json on 2026-10-28.
zenodo 17487080 data/raw/urbino olivetti.csv turing.csv von_neumann.csv

# DALTON (GitHub, AGPL-3.0; non-commercial research). CSVs are in Git LFS: requires `git lfs install` once.
command -v git-lfs >/dev/null || { echo "git lfs is required for DALTON (install it, then run: git lfs install)"; exit 1; }
echo "DALTON is ~1,335 CSVs at 1 Hz (several GB); free disk space:"; df -h . | tail -1
if [ ! -d data/raw/dalton/.git ]; then
  git clone --depth 1 https://github.com/prasenjit52282/dalton-dataset data/raw/dalton
fi
git -C data/raw/dalton lfs pull
rm -rf data/raw/dalton/.git/lfs/objects  # LFS cache duplicates the ~13 GB checkout; re-pull if ever needed

ls -l data/raw/*
