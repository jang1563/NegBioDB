#!/bin/bash
#SBATCH --job-name=ge_baseline
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/ge_baseline_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/ge_baseline_%j.err

# Train GE ML baseline model.
# Usage:
#   sbatch --export=ALL,TASK=m1,MODEL=xgboost,SPLIT=random,NEG=negbiodb,SEED=42 slurm/train_ge_baseline.slurm

set -euo pipefail

SCRATCH="/athena/masonlab/scratch/users/jak4013"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

TASK="${TASK:-m1}"
MODEL="${MODEL:-xgboost}"
SPLIT="${SPLIT:-random}"
NEG="${NEG:-negbiodb}"
SEED="${SEED:-42}"
BALANCED="${BALANCED:-}"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONPATH="${PROJECT_DIR}/src"

echo "=== GE Baseline Training ==="
echo "Task: ${TASK}, Model: ${MODEL}, Split: ${SPLIT}, Neg: ${NEG}, Seed: ${SEED}"
echo "Node: $(hostname)"
echo "Start: $(date)"

BALANCED_FLAG=""
if [[ -n "${BALANCED}" ]]; then
    BALANCED_FLAG="--balanced"
fi

python "${PROJECT_DIR}/scripts_depmap/train_ge_baseline.py" \
    --db-path "${PROJECT_DIR}/data/negbiodb_depmap.db" \
    --task "${TASK}" \
    --split "${SPLIT}" \
    --neg-source "${NEG}" \
    --model "${MODEL}" \
    --seed "${SEED}" \
    --gene-effect-file "${PROJECT_DIR}/data/depmap_raw/CRISPRGeneEffect.csv" \
    --dependency-file "${PROJECT_DIR}/data/depmap_raw/CRISPRGeneDependency.csv" \
    --output-dir "${PROJECT_DIR}/results/ge" \
    ${BALANCED_FLAG}

echo "End: $(date)"
