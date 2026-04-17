#!/bin/bash
#SBATCH --job-name=dc-train-cpu
#SBATCH --partition=scu-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/dc_train_cpu_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/dc_train_cpu_%j.err

# Train DC baselines (XGBoost/MLP) — CPU only
# Usage: sbatch --export=ALL,MODEL=xgboost,TASK=m1,SPLIT=random,SEED=42 slurm/run_dc_train_cpu.slurm

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

MODEL="${MODEL:-xgboost}"
TASK="${TASK:-m1}"
SPLIT="${SPLIT:-random}"
SEED="${SEED:-42}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/exports/dc_ml}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/results/dc_baselines}"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONPATH="${PROJECT_DIR}/src"

echo "=== DC CPU Training: model=${MODEL} task=${TASK} split=${SPLIT} seed=${SEED} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"

python "${PROJECT_DIR}/scripts_dc/train_dc_baseline.py" \
    --model "${MODEL}" \
    --task "${TASK}" \
    --split "${SPLIT}" \
    --seed "${SEED}" \
    --device cpu \
    --data-dir "${DATA_DIR}" \
    --output-dir "${RESULTS_DIR}"

echo "=== Done: $(date) ==="
