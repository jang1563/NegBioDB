#!/bin/bash
#SBATCH --job-name=rl_dpo
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/rl_dpo_%j.log
#SBATCH --error=logs/rl_dpo_%j.err

# === Resolve project root ===
cd "${SLURM_SUBMIT_DIR:-.}"

# === Configuration (override via sbatch --export=ALL,...) ===
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CONFIG="${CONFIG:-configs/negbiorl/dpo_qwen25_7b.yaml}"
SFT_ADAPTER="${SFT_ADAPTER:-}"
DATASET="${DATASET:-results/negbiorl/phase2_training_data/dpo_pairs.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/negbiorl/phase3_checkpoints/dpo_${SLURM_JOB_ID}}"

SCRATCH="/athena/masonlab/scratch/users/jak4013"
CONDA_ENV="${SCRATCH}/conda_env/negbiorl-train"

# === Setup ===
set -euo pipefail
mkdir -p logs

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config file not found: ${CONFIG}" >&2
    exit 1
fi

echo "=== NegBioRL DPO Training ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Model: ${BASE_MODEL}"
echo "SFT adapter: ${SFT_ADAPTER:-none}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

source /home/fs01/jak4013/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
export PYTHONPATH=src:${PYTHONPATH:-}
export PYTORCH_ALLOC_CONF=expandable_segments:True

# === Build command ===
CMD="accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    scripts_rl/06b_train_dpo.py \
    --base-model ${BASE_MODEL} \
    --dataset ${DATASET} \
    --config ${CONFIG} \
    --output-dir ${OUTPUT_DIR}"

if [ -n "${SFT_ADAPTER}" ]; then
    CMD="${CMD} --sft-adapter ${SFT_ADAPTER}"
fi

# === Run ===
eval ${CMD}

echo "=== Done: ${OUTPUT_DIR} ==="
