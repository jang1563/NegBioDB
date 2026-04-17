#!/bin/bash
#SBATCH --job-name=rl_grpo
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/rl_grpo_%j.log
#SBATCH --error=logs/rl_grpo_%j.err

# === Resolve project root ===
cd "${SLURM_SUBMIT_DIR:-.}"

# === Configuration (override via sbatch --export=ALL,...) ===
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
CONFIG="${CONFIG:-configs/negbiorl/grpo_qwen3_8b.yaml}"
SFT_ADAPTER="${SFT_ADAPTER:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
NUM_GEN="${NUM_GEN:-8}"
DATASET_WAS_EXPLICIT=0
if [ -n "${DATASET+x}" ]; then
    DATASET_WAS_EXPLICIT=1
fi

SCRATCH="/athena/masonlab/scratch/users/jak4013"
CONDA_ENV="${SCRATCH}/conda_env/negbiorl-train"
EPOCHS="${EPOCHS:-2}"
DATASET="${DATASET:-results/negbiorl/phase2_training_data/grpo_dataset.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/negbiorl/phase3_checkpoints/grpo_G${NUM_GEN}_${SLURM_JOB_ID}}"

# === Setup ===
set -euo pipefail
mkdir -p logs

# Validate CONFIG exists
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config file not found: ${CONFIG}" >&2
    echo "  (cwd=$(pwd), expected relative to project root)" >&2
    exit 1
fi

if [ ! -f "${DATASET}" ]; then
    echo "ERROR: Dataset file not found: ${DATASET}" >&2
    exit 1
fi

if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    if [ "${DATASET_WAS_EXPLICIT}" -ne 1 ]; then
        echo "ERROR: RESUME_FROM_CHECKPOINT requires DATASET to be explicitly set via sbatch --export." >&2
        echo "  Refusing to resume with implicit default dataset: ${DATASET}" >&2
        exit 1
    fi
    if [ ! -d "${RESUME_FROM_CHECKPOINT}" ]; then
        echo "ERROR: Resume checkpoint not found: ${RESUME_FROM_CHECKPOINT}" >&2
        exit 1
    fi
fi

echo "=== NegBioRL GRPO Training ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Model: ${BASE_MODEL}"
echo "G: ${NUM_GEN}"
echo "SFT adapter: ${SFT_ADAPTER:-none}"
echo "Dataset: ${DATASET}"
echo "Resume checkpoint: ${RESUME_FROM_CHECKPOINT:-none}"
echo "Output dir: ${OUTPUT_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

source /home/fs01/jak4013/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
export PYTHONPATH=src:${PYTHONPATH:-}

# === Build command ===
CMD="accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    scripts_rl/06_train_grpo.py \
    --base-model ${BASE_MODEL} \
    --dataset ${DATASET} \
    --config ${CONFIG} \
    --output-dir ${OUTPUT_DIR} \
    --num-generations ${NUM_GEN} \
    --epochs ${EPOCHS}"

if [ -n "${SFT_ADAPTER}" ]; then
    CMD="${CMD} --sft-adapter ${SFT_ADAPTER}"
fi

if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    CMD="${CMD} --resume-from-checkpoint ${RESUME_FROM_CHECKPOINT}"
fi

# === Run ===
eval ${CMD}

echo "=== Done: ${OUTPUT_DIR} ==="
