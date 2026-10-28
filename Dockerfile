# Edge-simulation image: the same code, run under docker --cpus / --memory limits.
# Build:  docker build -t coldstart .
# Data is NOT baked in: mount data/raw read-only (see scripts/run_edge_matrix.sh).
FROM python:3.12-slim

# libgomp1: OpenMP runtime needed by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -e ".[models]"
COPY scripts ./scripts
COPY config.yaml ./

# Thread pools must match the CPU limit, or BLAS/OpenMP oversubscribe and timings get noisy.
# run_edge_matrix.sh overrides these per configuration.
ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "scripts/run_learning_curve.py"]
