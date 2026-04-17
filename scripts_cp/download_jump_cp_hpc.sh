#!/usr/bin/env bash
set -euo pipefail

# Mirror JUMP Cell Painting assets on HPC instead of local storage.
#
# Usage:
#   bash scripts_cp/download_jump_cp_hpc.sh /path/on/hpc/cpg0016-jump
#   INCLUDE_RAW_IMAGES=1 bash scripts_cp/download_jump_cp_hpc.sh /path/on/hpc/cpg0016-jump

TARGET_ROOT="${1:?Provide a target directory on HPC}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_URI="${SOURCE_URI:-s3://cellpainting-gallery/cpg0016-jump}"
INCLUDE_RAW_IMAGES="${INCLUDE_RAW_IMAGES:-0}"

mkdir -p "${TARGET_ROOT}"

echo "Syncing JUMP Cell Painting assets to ${TARGET_ROOT}"
if command -v aws >/dev/null 2>&1; then
  aws s3 sync --no-sign-request "${SOURCE_URI}/workspace" "${TARGET_ROOT}/workspace"
  aws s3 sync --no-sign-request "${SOURCE_URI}/workspace_dl" "${TARGET_ROOT}/workspace_dl"

  if [[ "${INCLUDE_RAW_IMAGES}" == "1" ]]; then
    echo "Syncing raw images to ${TARGET_ROOT}/images"
    aws s3 sync --no-sign-request "${SOURCE_URI}/images" "${TARGET_ROOT}/images"
  else
    echo "Skipping raw images. Set INCLUDE_RAW_IMAGES=1 to mirror them on HPC."
  fi
fi

echo "Refreshing GitHub-hosted normalized metadata tables"
if command -v aws >/dev/null 2>&1; then
  python "${PROJECT_ROOT}/scripts_cp/download_jump_cp_https.py" --target-root "${TARGET_ROOT}" --skip-github-metadata
  mkdir -p "${TARGET_ROOT}/metadata"
  for file in README.md plate.csv.gz well.csv.gz compound.csv.gz compound_source.csv.gz perturbation_control.csv; do
    curl -L --fail --output "${TARGET_ROOT}/metadata/${file}" "https://raw.githubusercontent.com/jump-cellpainting/datasets/main/metadata/${file}"
  done
else
  echo "aws not found; falling back to HTTPS manifest-driven download"
  HTTPS_ARGS=(--target-root "${TARGET_ROOT}")
  if [[ "${INCLUDE_RAW_IMAGES}" == "1" ]]; then
    HTTPS_ARGS+=(--include-raw-images)
  fi
  python "${PROJECT_ROOT}/scripts_cp/download_jump_cp_https.py" "${HTTPS_ARGS[@]}"
fi

echo "Done. Production ingest prefers ${TARGET_ROOT}/metadata/*.csv.gz plus plate/backend assets."
