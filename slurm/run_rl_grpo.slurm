#!/bin/bash
#SBATCH --job-name=rl_grpo
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/rl_grpo_%j.log
#SBATCH --error=logs/rl_grpo_%j.err

# === Configuration (override via sbatch --export) ===
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
CONFIG="${CONFIG:-configs/negbiorl/grpo_qwen3_8b.yaml}"
SFT_ADAPTER="${SFT_ADAPTER:-}"
NUM_GEN="${NUM_GEN:-8}"

SCRATCH="/athena/masonlab/scratch/users/jak4013"
CONDA_ENV="${SCRATCH}/conda_env/negbiorl-train"
DATASET="results/negbiorl/phase2_training_data/grpo_dataset.jsonl"
OUTPUT_DIR="results/negbiorl/phase3_checkpoints/grpo_G${NUM_GEN}_${SLURM_JOB_ID}"

# === Setup ===
set -euo pipefail
mkdir -p logs

echo "=== NegBioRL GRPO Training ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Model: ${BASE_MODEL}"
echo "G: ${NUM_GEN}"
echo "SFT adapter: ${SFT_ADAPTER:-none}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

source activate "${CONDA_ENV}"
export PYTHONPATH=src:${PYTHONPATH:-}

# === Build command ===
CMD="accelerate launch scripts_rl/06_train_grpo.py \
    --base-model ${BASE_MODEL} \
    --dataset ${DATASET} \
    --config ${CONFIG} \
    --output-dir ${OUTPUT_DIR} \
    --num-generations ${NUM_GEN}"

if [ -n "${SFT_ADAPTER}" ]; then
    CMD="${CMD} --sft-adapter ${SFT_ADAPTER}"
fi

# === Run ===
eval ${CMD}

echo "=== Done: ${OUTPUT_DIR} ==="
