#!/bin/bash
#SBATCH --job-name=dc-train-gpu
#SBATCH --partition=scu-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/dc_train_gpu_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/dc_train_gpu_%j.err

# Train DC baselines (DeepSynergy/DrugCombGNN) — GPU required
# Usage: sbatch --export=ALL,MODEL=deepsynergy,TASK=m1,SPLIT=random,SEED=42 slurm/run_dc_train_gpu.slurm

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

MODEL="${MODEL:-deepsynergy}"
TASK="${TASK:-m1}"
SPLIT="${SPLIT:-random}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-50}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/exports/dc_ml}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/results/dc_baselines}"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONPATH="${PROJECT_DIR}/src"
_NVIDIA_LIBS="$(python -c 'import nvidia, os; print(os.path.dirname(nvidia.__file__))' 2>/dev/null || echo "")"
if [[ -n "${_NVIDIA_LIBS}" ]]; then
    for _d in "${_NVIDIA_LIBS}"/*/lib; do
        [ -d "${_d}" ] && export LD_LIBRARY_PATH="${_d}:${LD_LIBRARY_PATH:-}"
    done
fi

echo "=== DC GPU Training: model=${MODEL} task=${TASK} split=${SPLIT} seed=${SEED} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

python "${PROJECT_DIR}/scripts_dc/train_dc_baseline.py" \
    --model "${MODEL}" \
    --task "${TASK}" \
    --split "${SPLIT}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --device cuda \
    --data-dir "${DATA_DIR}" \
    --output-dir "${RESULTS_DIR}"

echo "=== Done: $(date) ==="
