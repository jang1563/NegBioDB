#!/bin/bash
#SBATCH --job-name=dc_synergy
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/dc_synergy_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/dc_synergy_%j.err

# Batch compute synergy scores from DrugComb dose-response data using R SynergyFinder.
# Usage:
#   sbatch slurm/run_dc_compute_synergy.slurm
#   sbatch --export=ALL,USE_PYTHON=1 slurm/run_dc_compute_synergy.slurm  # Python fallback

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"
USE_PYTHON="${USE_PYTHON:-0}"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONPATH="${PROJECT_DIR}/src"

echo "=== DC Synergy Score Computation ==="
echo "Project dir: ${PROJECT_DIR}"
echo "Use Python fallback: ${USE_PYTHON}"
date

CSV="${PROJECT_DIR}/data/dc/drugcomb/drugcomb_data_v1.4.csv"
OUTPUT="${PROJECT_DIR}/data/dc/synergy_scores.parquet"

if [[ ! -f "${CSV}" ]]; then
    echo "ERROR: DrugComb CSV not found: ${CSV}"
    exit 1
fi

EXTRA_ARGS=""
if [[ "${USE_PYTHON}" == "1" ]]; then
    EXTRA_ARGS="--use-python-fallback"
fi

python "${PROJECT_DIR}/scripts_dc/compute_synergy_scores.py" \
    --csv "${CSV}" \
    --output "${OUTPUT}" \
    ${EXTRA_ARGS}

echo "=== Done ==="
date
