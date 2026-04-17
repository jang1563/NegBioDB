#!/usr/bin/env bash
set -euo pipefail

# Run a plumbing-only proxy Cell Painting smoke on HPC using one JUMP plate and
# an assembled metadata fallback.
#
# Usage:
#   bash scripts_cp/run_jump_cp_hpc_smoke.sh
#   SCRATCH_ROOT=/athena/masonlab/scratch/users/$USER/negbiodb_cp_smoke bash scripts_cp/run_jump_cp_hpc_smoke.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/fs01/jak4013/miniconda3/miniconda3/envs/negbiodb-ml/bin/python}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/athena/masonlab/scratch/users/jak4013/negbiodb_cp_proxy_smoke}"
DATA_ROOT="${SCRATCH_ROOT}/data"
WORK_ROOT="${SCRATCH_ROOT}/work"
EXPORT_ROOT="${SCRATCH_ROOT}/exports/cp_ml"
DB_PATH="${SCRATCH_ROOT}/data/negbiodb_cp_proxy_smoke.db"

ASSEMBLED_URL="https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump-assembled/source_all/workspace/profiles_assembled/COMPOUND/v1.0/profiles_var_mad_int_featselect_harmony.parquet"
PLATE_PROFILE_URL="https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_4/workspace/profiles/2021_04_26_Batch1/BR00117035/BR00117035.parquet"
BACKEND_CSV_URL="https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_4/workspace/backend/2021_04_26_Batch1/BR00117035/BR00117035.csv"

mkdir -p "${DATA_ROOT}" "${WORK_ROOT}" "${EXPORT_ROOT}" "${SCRATCH_ROOT}/results"

download_if_missing() {
  local url="$1"
  local out="$2"
  if [[ -f "$out" ]]; then
    echo "Using cached file: $out"
    return
  fi
  echo "Downloading: $url"
  curl -L --fail --output "$out" "$url"
}

download_if_missing "$ASSEMBLED_URL" "${DATA_ROOT}/profiles_var_mad_int_featselect_harmony.parquet"
download_if_missing "$PLATE_PROFILE_URL" "${DATA_ROOT}/BR00117035.parquet"
download_if_missing "$BACKEND_CSV_URL" "${DATA_ROOT}/BR00117035.csv"

"$PYTHON_BIN" "${PROJECT_ROOT}/scripts_cp/prepare_jump_plate_smoke.py" \
  --assembled-compound-parquet "${DATA_ROOT}/profiles_var_mad_int_featselect_harmony.parquet" \
  --plate-profile-parquet "${DATA_ROOT}/BR00117035.parquet" \
  --backend-csv "${DATA_ROOT}/BR00117035.csv" \
  --output-observations "${WORK_ROOT}/jump_proxy_smoke_observations.parquet" \
  --output-profile-features "${WORK_ROOT}/jump_proxy_smoke_profile_features.parquet" \
  --output-meta "${WORK_ROOT}/jump_proxy_smoke_meta.json" \
  --batch-name "2021_04_26_Batch1"

cd "$PROJECT_ROOT"

PYTHONPATH=src "$PYTHON_BIN" scripts_cp/load_jump_cp.py \
  --observations "${WORK_ROOT}/jump_proxy_smoke_observations.parquet" \
  --profile-features "${WORK_ROOT}/jump_proxy_smoke_profile_features.parquet" \
  --db-path "${DB_PATH}" \
  --annotation-mode plate_proxy \
  --dataset-name "cpg0016-jump-proxy-smoke" \
  --dataset-version "proxy_smoke_2026-04-10"

PYTHONPATH=src "$PYTHON_BIN" scripts_cp/export_cp_ml_dataset.py \
  --db-path "${DB_PATH}" \
  --allow-proxy-smoke \
  --output-dir "${EXPORT_ROOT}"

echo "Proxy smoke DB: ${DB_PATH}"
echo "Proxy smoke exports: ${EXPORT_ROOT}"
echo "Proxy smoke metadata: ${WORK_ROOT}/jump_proxy_smoke_meta.json"
