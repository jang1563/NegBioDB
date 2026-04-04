#!/bin/bash
#SBATCH --job-name=vp_esm2
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/vp_esm2_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/vp_esm2_%j.err

# Precompute ESM2 embeddings for missense VP variants.
# Usage:
#   sbatch --export=ALL slurm/run_vp_precompute_esm2.slurm

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

DB_PATH="${DB_PATH:-${PROJECT_DIR}/data/negbiodb_vp.db}"
SEQUENCES="${SEQUENCES:-${PROJECT_DIR}/data/vp/protein_sequences.tsv}"
OUTPUT="${OUTPUT:-${PROJECT_DIR}/data/vp/esm2_embeddings.parquet}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-1024}"

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

echo "=== VP ESM2 Precompute ==="
echo "DB: ${DB_PATH}"
echo "Sequences: ${SEQUENCES}"
echo "Output: ${OUTPUT}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi 2>&1 || echo "nvidia-smi failed"
echo "Start: $(date)"

python "${PROJECT_DIR}/scripts_vp/precompute_esm2.py" \
    --db "${DB_PATH}" \
    --sequences "${SEQUENCES}" \
    --output "${OUTPUT}" \
    --device cuda \
    --batch-size "${BATCH_SIZE}" \
    --max-length "${MAX_LENGTH}"

echo "End: $(date)"
