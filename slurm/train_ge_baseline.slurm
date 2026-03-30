#!/bin/bash
#SBATCH --job-name=ge_baseline
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=${SCRATCH_DIR:-/path/to/scratch}/negbiodb/logs/ge_baseline_%j.log
#SBATCH --error=${SCRATCH_DIR:-/path/to/scratch}/negbiodb/logs/ge_baseline_%j.err

# Train GE ML baseline model.
# Usage:
#   sbatch --export=ALL,TASK=m1,MODEL=xgboost,SPLIT=random,NEG=negbiodb,SEED=42 slurm/train_ge_baseline.slurm

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/path/to/scratch}"
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
# Provide CUDA libs (cudnn, cublas) for PyTorch on CPU-only nodes
_NVIDIA_LIBS="${CONDA_PREFIX:-/path/to/conda}/miniconda3/envs/negbiodb-llm/lib/python3.11/site-packages/nvidia"
for _d in "${_NVIDIA_LIBS}"/*/lib; do
    [ -d "${_d}" ] && export LD_LIBRARY_PATH="${_d}:${LD_LIBRARY_PATH:-}"
done

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
