#!/usr/bin/env bash
set -euo pipefail

# Ingest one JUMP Cell Painting batch (production-grade, annotation-backed).
#
# Downloads metadata + plate data from S3 via HTTPS (no aws CLI needed),
# runs prepare_jump_plate_production.py per plate, then loads into CP DB.
#
# Usage:
#   bash scripts_cp/ingest_jump_production.sh
#   SOURCE=source_4 BATCH=2021_04_26_Batch1 bash scripts_cp/ingest_jump_production.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fs01/jak4013/miniconda3/miniconda3/envs/negbiodb-ml/bin/python}"
SCRATCH="${SCRATCH:-/athena/masonlab/scratch/users/jak4013}"

SOURCE="${SOURCE:-source_4}"
BATCH="${BATCH:-2021_04_26_Batch1}"
S3_BASE="https://cellpainting-gallery.s3.amazonaws.com"
S3_PREFIX="cpg0016-jump/${SOURCE}"

MIRROR_ROOT="${SCRATCH}/jump_mirror/cpg0016-jump/${SOURCE}"
WORK_ROOT="${SCRATCH}/negbiodb_cp_production/work/${SOURCE}/${BATCH}"
DB_PATH="${SCRATCH}/negbiodb/data/negbiodb_cp.db"

mkdir -p "${MIRROR_ROOT}/workspace/structure" "${WORK_ROOT}" "$(dirname "${DB_PATH}")"

# ---------- helpers ----------
download_if_missing() {
  local url="$1" out="$2"
  if [[ -f "$out" ]]; then
    echo "[cached] $out"
    return
  fi
  mkdir -p "$(dirname "$out")"
  echo "[downloading] $url"
  curl -L --fail --retry 3 --connect-timeout 30 -o "$out" "$url"
}

s3_list_keys() {
  # List all keys under an S3 prefix using the ListBucketV2 API
  local prefix="$1"
  local token=""
  local url
  local token_file
  token_file=$(mktemp /tmp/_s3_token_XXXXXX)
  trap "rm -f '$token_file'" RETURN
  while true; do
    if [[ -z "$token" ]]; then
      url="${S3_BASE}?list-type=2&prefix=${prefix}&max-keys=1000"
    else
      local encoded_token
      encoded_token=$(python3 -c "import urllib.parse, sys; print(urllib.parse.quote(sys.argv[1], safe=''))" "$token")
      url="${S3_BASE}?list-type=2&prefix=${prefix}&max-keys=1000&continuation-token=${encoded_token}"
    fi
    local xml
    xml=$(curl -s --connect-timeout 30 "$url")
    echo "$xml" | python3 -c "
import sys, xml.etree.ElementTree as ET
ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
root = ET.fromstring(sys.stdin.read())
for k in root.findall('s3:Contents/s3:Key', ns):
    print(k.text)
trunc = root.findtext('s3:IsTruncated', namespaces=ns)
nxt = root.findtext('s3:NextContinuationToken', namespaces=ns)
if trunc == 'true' and nxt:
    print(nxt, file=sys.stderr)
" 2>"$token_file"
    token=$(cat "$token_file" 2>/dev/null || true)
    [[ -z "$token" ]] && break
  done
}

# ---------- Step 1: Download structure.json ----------
echo "=== Step 1: structure.json ==="
download_if_missing \
  "${S3_BASE}/${S3_PREFIX}/workspace/structure/structure.json" \
  "${MIRROR_ROOT}/workspace/structure/structure.json"

# ---------- Step 2: Find and download metadata files ----------
echo "=== Step 2: Metadata files ==="

# List metadata directory to find barcode_platemap and platemaps
META_PREFIX="${S3_PREFIX}/workspace/metadata/"
echo "Listing ${META_PREFIX}..."
META_KEYS=$(s3_list_keys "${META_PREFIX}" 2>/dev/null || true)

if [[ -z "$META_KEYS" ]]; then
  # Try alternative metadata locations
  for alt_prefix in \
    "${S3_PREFIX}/workspace/load_data_csv/${BATCH}/" \
    "cpg0016-jump/source_all/workspace/metadata/" \
    "cpg0016-jump-assembled/source_all/workspace/metadata/"; do
    echo "Trying alternative: ${alt_prefix}..."
    META_KEYS=$(s3_list_keys "${alt_prefix}" 2>/dev/null || true)
    [[ -n "$META_KEYS" ]] && break
  done
fi

if [[ -n "$META_KEYS" ]]; then
  echo "$META_KEYS" | while read -r key; do
    [[ -z "$key" || "$key" == */ ]] && continue
    download_if_missing "${S3_BASE}/${key}" "${SCRATCH}/jump_mirror/${key}"
  done
else
  echo "WARNING: No metadata keys found. Will attempt production ingest anyway."
fi

# Also try to find external_metadata from common locations
for ext_path in \
  "cpg0016-jump/${SOURCE}/workspace/metadata/external_metadata/JUMP-Target-2_compound_metadata.tsv" \
  "cpg0016-jump/${SOURCE}/workspace/metadata/external_metadata.tsv" \
  "cpg0016-jump/source_all/workspace/metadata/external_metadata.tsv"; do
  if curl -sf --head "${S3_BASE}/${ext_path}" >/dev/null 2>&1; then
    download_if_missing "${S3_BASE}/${ext_path}" "${SCRATCH}/jump_mirror/${ext_path}"
    break
  fi
done

# ---------- Step 3: Find plates in this batch ----------
echo "=== Step 3: Listing plates for ${BATCH} ==="
PROFILE_PREFIX="${S3_PREFIX}/workspace/profiles/${BATCH}/"
PLATE_DIRS=$(s3_list_keys "${PROFILE_PREFIX}" 2>/dev/null | grep -oP "profiles/${BATCH}/\K[^/]+" | sort -u || true)

if [[ -z "$PLATE_DIRS" ]]; then
  echo "ERROR: No plates found at ${PROFILE_PREFIX}"
  echo "Falling back to single known plate: BR00117035"
  PLATE_DIRS="BR00117035"
fi

PLATE_COUNT=$(echo "$PLATE_DIRS" | wc -l | tr -d ' ')
echo "Found ${PLATE_COUNT} plates"

# Limit to first N plates for initial validation
MAX_PLATES="${MAX_PLATES:-5}"
PLATE_DIRS=$(echo "$PLATE_DIRS" | head -n "${MAX_PLATES}")
echo "Processing first ${MAX_PLATES} plates: $(echo $PLATE_DIRS | tr '\n' ' ')"

# ---------- Step 4: Download plate data + prepare ----------
echo "=== Step 4: Download + prepare plates ==="
OBS_FILES=()

for PLATE in $PLATE_DIRS; do
  echo "--- Plate: ${PLATE} ---"
  PLATE_DIR="${SCRATCH}/jump_mirror/${S3_PREFIX}/workspace"

  # Download profile parquet
  download_if_missing \
    "${S3_BASE}/${S3_PREFIX}/workspace/profiles/${BATCH}/${PLATE}/${PLATE}.parquet" \
    "${PLATE_DIR}/profiles/${BATCH}/${PLATE}/${PLATE}.parquet"

  # Download backend CSV
  download_if_missing \
    "${S3_BASE}/${S3_PREFIX}/workspace/backend/${BATCH}/${PLATE}/${PLATE}.csv" \
    "${PLATE_DIR}/backend/${BATCH}/${PLATE}/${PLATE}.csv"

  # Run production preparation
  OBS_OUT="${WORK_ROOT}/${PLATE}_observations.parquet"
  PROF_OUT="${WORK_ROOT}/${PLATE}_profile_features.parquet"
  META_OUT="${WORK_ROOT}/${PLATE}_meta.json"

  if [[ -f "$OBS_OUT" ]]; then
    echo "[cached] ${OBS_OUT}"
  else
    cd "${PROJECT_ROOT}"
    PYTHONPATH=src ${PYTHON} scripts_cp/prepare_jump_plate_production.py \
      --metadata-root "${SCRATCH}/jump_mirror/cpg0016-jump/${SOURCE}" \
      --batch-name "${BATCH}" \
      --plate-name "${PLATE}" \
      --source-name "${SOURCE}" \
      --plate-profile-parquet "${PLATE_DIR}/profiles/${BATCH}/${PLATE}/${PLATE}.parquet" \
      --backend-csv "${PLATE_DIR}/backend/${BATCH}/${PLATE}/${PLATE}.csv" \
      --output-observations "${OBS_OUT}" \
      --output-profile-features "${PROF_OUT}" \
      --output-meta "${META_OUT}" \
      --cell-line-name U2OS \
      --timepoint-h 48.0 || {
        echo "WARNING: prepare failed for plate ${PLATE}, skipping"
        continue
      }
  fi

  OBS_FILES+=("${OBS_OUT}")
done

# ---------- Step 5: Concatenate and load ----------
echo "=== Step 5: Concatenate and load into DB ==="
cd "${PROJECT_ROOT}"

CONCAT_OBS="${WORK_ROOT}/batch_observations.parquet"
CONCAT_PROF="${WORK_ROOT}/batch_profile_features.parquet"

PYTHONPATH=src ${PYTHON} -c "
import pandas as pd
import glob
obs_files = sorted(glob.glob('${WORK_ROOT}/*_observations.parquet'))
prof_files = sorted(glob.glob('${WORK_ROOT}/*_profile_features.parquet'))
if not obs_files:
    raise RuntimeError('No observation files found')
obs = pd.concat([pd.read_parquet(f) for f in obs_files], ignore_index=True)
prof = pd.concat([pd.read_parquet(f) for f in prof_files], ignore_index=True)
obs.to_parquet('${CONCAT_OBS}', index=False)
prof.to_parquet('${CONCAT_PROF}', index=False)
print(f'Concatenated {len(obs_files)} plates: {len(obs)} observations, {len(prof)} profiles')
"

PYTHONPATH=src ${PYTHON} scripts_cp/load_jump_cp.py \
  --db-path "${DB_PATH}" \
  --observations "${CONCAT_OBS}" \
  --profile-features "${CONCAT_PROF}" \
  --dataset-name "cpg0016-jump" \
  --dataset-version "1.0" \
  --annotation-mode annotated \
  --source-url "${S3_BASE}/${S3_PREFIX}"

echo "=== Done ==="
echo "DB: ${DB_PATH}"
echo "Observations: ${CONCAT_OBS}"
echo "Profiles: ${CONCAT_PROF}"
ls -lh "${DB_PATH}"
