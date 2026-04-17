#!/usr/bin/env bash
set -euo pipefail

# Run an annotation-backed Cell Painting production smoke on HPC using one real
# JUMP compound plate plus the normalized GitHub metadata tables.
#
# Usage:
#   bash scripts_cp/run_jump_cp_production_smoke.sh
#   SOURCE_NAME=source_2 BATCH_NAME=20210607_Batch_2 PLATE_NAME=1053601756 \
#     bash scripts_cp/run_jump_cp_production_smoke.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/fs01/jak4013/miniconda3/miniconda3/envs/negbiodb-ml/bin/python}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/athena/masonlab/scratch/users/jak4013/negbiodb_cp_prod_smoke}"
SOURCE_NAME="${SOURCE_NAME:-source_1}"
BATCH_NAME="${BATCH_NAME:-Batch1_20221004}"
PLATE_NAME="${PLATE_NAME:-UL001641}"
CELL_LINE_NAME="${CELL_LINE_NAME:-U2OS}"
DEFAULT_COMPOUND_DOSE="${DEFAULT_COMPOUND_DOSE:-10.0}"

RUN_SLUG="${SOURCE_NAME}_${PLATE_NAME}"
DATA_ROOT="${SCRATCH_ROOT}/data/${RUN_SLUG}"
META_ROOT="${SCRATCH_ROOT}/meta"
WORK_ROOT="${SCRATCH_ROOT}/work/${RUN_SLUG}"
EXPORT_ML_ROOT="${SCRATCH_ROOT}/exports/${RUN_SLUG}/cp_ml"
EXPORT_LLM_ROOT="${SCRATCH_ROOT}/exports/${RUN_SLUG}/cp_llm"
RESULTS_ROOT="${SCRATCH_ROOT}/results/${RUN_SLUG}"
DB_PATH="${SCRATCH_ROOT}/data/negbiodb_cp_prod_${RUN_SLUG}.db"

PLATE_PROFILE_URL="https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/${SOURCE_NAME}/workspace/profiles/${BATCH_NAME}/${PLATE_NAME}/${PLATE_NAME}.parquet"
BACKEND_CSV_URL="https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/${SOURCE_NAME}/workspace/backend/${BATCH_NAME}/${PLATE_NAME}/${PLATE_NAME}.csv"

mkdir -p "${DATA_ROOT}" "${META_ROOT}" "${WORK_ROOT}" "${EXPORT_ML_ROOT}" "${EXPORT_LLM_ROOT}" "${RESULTS_ROOT}"

download_if_missing() {
  local url="$1"
  local out="$2"
  mkdir -p "$(dirname "$out")"
  if [[ -f "$out" ]]; then
    echo "Using cached file: $out"
    return
  fi
  echo "Downloading: $url"
  curl -L --fail --output "$out" "$url"
}

for file in README.md plate.csv.gz well.csv.gz compound.csv.gz compound_source.csv.gz perturbation_control.csv; do
  download_if_missing \
    "https://raw.githubusercontent.com/jump-cellpainting/datasets/main/metadata/${file}" \
    "${META_ROOT}/metadata/${file}"
done

download_if_missing "${PLATE_PROFILE_URL}" "${DATA_ROOT}/${PLATE_NAME}.parquet"
download_if_missing "${BACKEND_CSV_URL}" "${DATA_ROOT}/${PLATE_NAME}.csv"

cd "${PROJECT_ROOT}"

PYTHONPATH=src "${PYTHON_BIN}" scripts_cp/prepare_jump_plate_production.py \
  --metadata-root "${META_ROOT}" \
  --batch-name "${BATCH_NAME}" \
  --plate-name "${PLATE_NAME}" \
  --source-name "${SOURCE_NAME}" \
  --cell-line-name "${CELL_LINE_NAME}" \
  --plate-profile-parquet "${DATA_ROOT}/${PLATE_NAME}.parquet" \
  --backend-csv "${DATA_ROOT}/${PLATE_NAME}.csv" \
  --output-observations "${WORK_ROOT}/observations.parquet" \
  --output-profile-features "${WORK_ROOT}/profile_features.parquet" \
  --output-meta "${WORK_ROOT}/production_meta.json" \
  --default-compound-dose "${DEFAULT_COMPOUND_DOSE}"

PYTHONPATH=src "${PYTHON_BIN}" scripts_cp/load_jump_cp.py \
  --observations "${WORK_ROOT}/observations.parquet" \
  --profile-features "${WORK_ROOT}/profile_features.parquet" \
  --db-path "${DB_PATH}" \
  --annotation-mode annotated \
  --dataset-name "cpg0016-jump-production-smoke" \
  --dataset-version "${RUN_SLUG}"

PYTHONPATH=src "${PYTHON_BIN}" scripts_cp/export_cp_ml_dataset.py \
  --db-path "${DB_PATH}" \
  --output-dir "${EXPORT_ML_ROOT}"

PYTHONPATH=src "${PYTHON_BIN}" scripts_cp/build_cp_l1_dataset.py \
  --db-path "${DB_PATH}" \
  --output-dir "${EXPORT_LLM_ROOT}" \
  --n-per-class 8 \
  --fewshot-per-class 2 \
  --val-per-class 0

PYTHONPATH=src "${PYTHON_BIN}" scripts_cp/build_cp_l4_dataset.py \
  --db-path "${DB_PATH}" \
  --output-dir "${EXPORT_LLM_ROOT}" \
  --n-per-class 40 \
  --fewshot-per-class 4 \
  --val-per-class 0

PYTHONPATH=src "${PYTHON_BIN}" scripts_cp/train_cp_baseline.py \
  --model mlp \
  --task m1 \
  --feature-set profile \
  --split random \
  --data-dir "${EXPORT_ML_ROOT}" \
  --output-dir "${RESULTS_ROOT}" \
  --smoke-test

echo "Production smoke DB: ${DB_PATH}"
echo "Production smoke ML exports: ${EXPORT_ML_ROOT}"
echo "Production smoke LLM exports: ${EXPORT_LLM_ROOT}"
echo "Production smoke results: ${RESULTS_ROOT}"
echo "Production smoke metadata: ${WORK_ROOT}/production_meta.json"
