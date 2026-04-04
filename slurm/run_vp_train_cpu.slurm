#!/bin/bash
#SBATCH --job-name=vp_train_cpu
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/vp_train_cpu_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/vp_train_cpu_%j.err

# Train VP CPU-based models (XGBoost, MLP).
# Usage:
#   sbatch --export=ALL,MODEL=xgboost,DATASET=m1_balanced,SPLIT=random,SEED=42 slurm/run_vp_train_cpu.slurm
#   sbatch --export=ALL,MODEL=mlp,DATASET=m1_realistic,SPLIT=cold_gene,SEED=42,EPOCHS=25 slurm/run_vp_train_cpu.slurm

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

MODEL="${MODEL:-xgboost}"
DATASET="${DATASET:-m1_balanced}"
SPLIT="${SPLIT:-random}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-50}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/exports/vp_ml}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/results/vp_baselines}"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONPATH="${PROJECT_DIR}/src"
_NVIDIA_LIBS="$(python -c 'import nvidia, os; print(os.path.dirname(nvidia.__file__))' 2>/dev/null || echo "")"
if [[ -n "${_NVIDIA_LIBS}" ]]; then
    for _d in "${_NVIDIA_LIBS}"/*/lib; do
        [ -d "${_d}" ] && export LD_LIBRARY_PATH="${_d}:${LD_LIBRARY_PATH:-}"
    done
    echo "Added nvidia libs from ${_NVIDIA_LIBS}"
fi

echo "=== VP CPU Training ==="
echo "Model: ${MODEL}, Dataset: ${DATASET}, Split: ${SPLIT}, Seed: ${SEED}"
echo "Node: $(hostname)"
echo "Start: $(date)"

python "${PROJECT_DIR}/scripts_vp/train_vp_baseline.py" \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --device cpu \
    --data-dir "${DATA_DIR}" \
    --output-dir "${RESULTS_DIR}"

echo "End: $(date)"
