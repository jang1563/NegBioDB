#!/bin/bash
#SBATCH --job-name=md-ingest
#SBATCH --partition=scu-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/md_ingest_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/md_ingest_%j.err

# MD domain full ingest pipeline (HMDB cache + MetaboLights + NMDR + aggregation)
# Usage: sbatch slurm/run_md_ingest.slurm

set -euo pipefail

SCRATCH="/athena/masonlab/scratch/users/jak4013"
PROJECT_DIR="${SCRATCH}/negbiodb"
SCRATCH_ENV_PYTHON="${SCRATCH}/conda_env/negbiodb-ml/bin/python"
HOME_ENV_PYTHON="/home/fs01/jak4013/miniconda3/miniconda3/envs/negbiodb-ml/bin/python"
PYTHON_BIN="${PYTHON_BIN:-}"
SKIP_HMDB="${SKIP_HMDB:-0}"

if [[ -z "${PYTHON_BIN}" ]]; then
    if [[ -x "${SCRATCH_ENV_PYTHON}" ]]; then
        PYTHON_BIN="${SCRATCH_ENV_PYTHON}"
    elif [[ -x "${HOME_ENV_PYTHON}" ]]; then
        PYTHON_BIN="${HOME_ENV_PYTHON}"
    else
        echo "ERROR: Could not find negbiodb-ml python interpreter."
        echo "Checked ${SCRATCH_ENV_PYTHON} and ${HOME_ENV_PYTHON}"
        exit 1
    fi
fi

export PYTHONPATH="${PROJECT_DIR}/src"

echo "=== MD Domain Ingest Pipeline ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start: $(date)"
echo "Python: ${PYTHON_BIN}"
echo "Skip HMDB: ${SKIP_HMDB}"

cd "${PROJECT_DIR}"

# Step 1: Download HMDB (internal cache only — CC BY-NC 4.0, not redistributed)
if [[ "${SKIP_HMDB}" == "1" ]]; then
    echo "--- Step 1: HMDB internal cache (skipped) ---"
else
    echo "--- Step 1: HMDB internal cache ---"
    "${PYTHON_BIN}" scripts_md/01_download_hmdb.py
fi

# Step 2: MetaboLights ingest
echo "--- Step 2: MetaboLights ---"
"${PYTHON_BIN}" scripts_md/02_ingest_metabolights.py

# Step 3: NMDR ingest
echo "--- Step 3: NMDR ---"
"${PYTHON_BIN}" scripts_md/03_ingest_nmdr.py

# Step 4: ClassyFire standardization
echo "--- Step 4: ClassyFire enrichment ---"
"${PYTHON_BIN}" scripts_md/04_standardize.py

# Step 5: Aggregate pairs + gold tier upgrade
echo "--- Step 5: Aggregate + gold upgrade ---"
"${PYTHON_BIN}" scripts_md/05_aggregate_pairs.py --upgrade-gold

echo "=== Done: $(date) ==="
