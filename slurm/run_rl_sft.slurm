#!/bin/bash
#SBATCH --job-name=rl_sft
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rl_sft_%j.log
#SBATCH --error=logs/rl_sft_%j.err

# === Configuration (override via sbatch --export) ===
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
CONFIG="${CONFIG:-configs/negbiorl/sft_qwen3_8b.yaml}"

SCRATCH="/athena/masonlab/scratch/users/jak4013"
CONDA_ENV="${SCRATCH}/conda_env/negbiorl-train"
DATASET="results/negbiorl/phase2_training_data/sft_dataset.jsonl"
OUTPUT_DIR="results/negbiorl/phase3_checkpoints/sft_${SLURM_JOB_ID}"

# === Setup ===
set -euo pipefail
mkdir -p logs

echo "=== NegBioRL SFT Training ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Model: ${BASE_MODEL}"
echo "Config: ${CONFIG}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

source /home/fs01/jak4013/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
export PYTHONPATH=src:${PYTHONPATH:-}

# === Run ===
accelerate launch scripts_rl/05_train_sft.py \
    --base-model "${BASE_MODEL}" \
    --dataset "${DATASET}" \
    --config "${CONFIG}" \
    --output-dir "${OUTPUT_DIR}"

echo "=== Done: ${OUTPUT_DIR} ==="
