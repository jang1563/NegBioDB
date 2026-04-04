#!/bin/bash
#SBATCH --job-name=vp_fetch_seq
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/vp_fetch_seq_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/vp_fetch_seq_%j.err

# Fetch reviewed human protein sequences from UniProt for VP missense genes.
# Usage:
#   sbatch --export=ALL slurm/run_vp_fetch_sequences.slurm

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

DB_PATH="${DB_PATH:-${PROJECT_DIR}/data/negbiodb_vp.db}"
OUTPUT="${OUTPUT:-${PROJECT_DIR}/data/vp/protein_sequences.tsv}"
BATCH_SIZE="${BATCH_SIZE:-100}"
DELAY="${DELAY:-0.5}"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONPATH="${PROJECT_DIR}/src"

echo "=== VP Protein Sequence Fetch ==="
echo "DB: ${DB_PATH}"
echo "Output: ${OUTPUT}"
echo "Node: $(hostname)"
echo "Start: $(date)"

python "${PROJECT_DIR}/scripts_vp/fetch_protein_sequences.py" \
    --db "${DB_PATH}" \
    --output "${OUTPUT}" \
    --batch-size "${BATCH_SIZE}" \
    --delay "${DELAY}"

echo "End: $(date)"
