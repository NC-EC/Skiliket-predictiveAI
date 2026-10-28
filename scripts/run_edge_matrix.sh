#!/usr/bin/env bash
# Time the learning curve under simulated edge limits: every combination of CPUs {1,2} x memory {1g,2g}.
# Run from the repo root on the timing-reference PC (Ryzen 5 5500), with nothing else heavy running.
#   docker build -t coldstart .
#   bash scripts/run_edge_matrix.sh dalton H1_41
# Extra runner arguments can follow the device, e.g. --days 1 3 7 --horizon 30.
# NOT YET RUN: written in the cloud session; check the first configuration's output before running all four.
set -euo pipefail
dataset=${1:?dataset}; device=${2:?device}; shift 2
case $dataset in
  dalton) path=/data/dalton ;; ediaqi) path=/data/ediaqi ;; urbino) path=/data/urbino ;;
  *) echo "unknown dataset $dataset"; exit 1 ;;
esac
mkdir -p results/edge
# Git Bash on Windows rewrites /paths; MSYS_NO_PATHCONV keeps container paths intact.
export MSYS_NO_PATHCONV=1
for cpus in 1 2; do
  for mem in 1g 2g; do
    tag="${dataset}_${device}_cpu${cpus}_mem${mem}"
    echo "== $tag"
    docker run --rm --cpus "$cpus" --memory "$mem" --memory-swap "$mem" \
      -e OMP_NUM_THREADS="$cpus" -e MKL_NUM_THREADS="$cpus" -e OPENBLAS_NUM_THREADS="$cpus" \
      -v "$(pwd)/data/raw:/data:ro" -v "$(pwd)/results/edge:/app/results/edge" \
      coldstart --dataset "$dataset" --path "$path" --device "$device" \
      --out "results/edge/${tag}.csv" "$@" \
      || echo "!! $tag failed (exit $?): out of memory shows as exit 137"
  done
done
