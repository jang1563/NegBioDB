#!/bin/bash
#SBATCH --job-name=vp_train_gpu
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/vp_train_gpu_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/vp_train_gpu_%j.err

# Train VP GPU-based models (ESM2-VP, VariantGNN).
# Usage:
#   sbatch --export=ALL,MODEL=esm2,DATASET=m1_balanced,SPLIT=random,SEED=42 slurm/run_vp_train_gpu.slurm
#   sbatch --export=ALL,MODEL=gnn,DATASET=m1_balanced,SPLIT=cold_gene,SEED=42 slurm/run_vp_train_gpu.slurm

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

MODEL="${MODEL:-esm2}"
DATASET="${DATASET:-m1_balanced}"
SPLIT="${SPLIT:-random}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-50}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/exports/vp_ml}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/results/vp_baselines}"
ESM2_EMBEDDINGS="${ESM2_EMBEDDINGS:-${PROJECT_DIR}/data/vp/esm2_embeddings.parquet}"
GENE_GRAPH="${GENE_GRAPH:-${PROJECT_DIR}/data/vp/string_gene_graph.pkl}"

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

echo "=== VP GPU Training ==="
echo "Model: ${MODEL}, Dataset: ${DATASET}, Split: ${SPLIT}, Seed: ${SEED}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi 2>&1 || echo "nvidia-smi failed"
echo "Start: $(date)"

cmd=(
    python "${PROJECT_DIR}/scripts_vp/train_vp_baseline.py"
    --model "${MODEL}"
    --dataset "${DATASET}"
    --split "${SPLIT}"
    --seed "${SEED}"
    --epochs "${EPOCHS}"
    --device cuda
    --data-dir "${DATA_DIR}"
    --output-dir "${RESULTS_DIR}"
)

if [ "${MODEL}" = "esm2" ]; then
    cmd+=(--esm2-embeddings "${ESM2_EMBEDDINGS}")
elif [ "${MODEL}" = "gnn" ]; then
    cmd+=(--gene-graph "${GENE_GRAPH}")
fi

"${cmd[@]}"

echo "End: $(date)"
