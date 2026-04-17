#!/bin/bash
#SBATCH --job-name=cp-train-cpu
#SBATCH --partition=scu-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/cp_train_cpu_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/cp_train_cpu_%j.err

# Train CP baselines (XGBoost/MLP) — CPU only
# Usage: sbatch --export=ALL,MODEL=xgboost,TASK=m1,FEATURE_SET=profile,SPLIT=random,SEED=42 slurm/run_cp_train_cpu.slurm

set -euo pipefail

SCRATCH="/athena/masonlab/scratch/users/jak4013"
PROJECT_DIR="${SCRATCH}/negbiodb"
PYTHON="/home/fs01/jak4013/miniconda3/miniconda3/envs/negbiodb-ml/bin/python"

MODEL="${MODEL:-xgboost}"
TASK="${TASK:-m1}"
FEATURE_SET="${FEATURE_SET:-profile}"
SPLIT="${SPLIT:-random}"
SEED="${SEED:-42}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/exports/cp_ml}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/results/cp_baselines}"

export PYTHONPATH="${PROJECT_DIR}/src"

echo "=== CP CPU Training: model=${MODEL} task=${TASK} features=${FEATURE_SET} split=${SPLIT} seed=${SEED} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start: $(date)"

cd "${PROJECT_DIR}"
${PYTHON} scripts_cp/train_cp_baseline.py \
    --model "${MODEL}" \
    --task "${TASK}" \
    --feature-set "${FEATURE_SET}" \
    --split "${SPLIT}" \
    --seed "${SEED}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${RESULTS_DIR}"

echo "=== Done: $(date) ==="
