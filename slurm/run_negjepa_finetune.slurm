#!/bin/bash
#SBATCH --job-name=negjepa_ft
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/negjepa_ft_%j.log
#SBATCH --error=logs/negjepa_ft_%j.err

# === Configuration (override via sbatch --export) ===
# Required:
PRETRAINED="${PRETRAINED:?PRETRAINED must be set: path to best.pt checkpoint}"
DOMAIN="${DOMAIN:?DOMAIN must be set: dti|ppi|ge|ct|vp|dc|cp|md}"
SPLIT="${SPLIT:-random}"
# Optional:
CONFIG="${CONFIG:-configs/negbiojepa/finetune.yaml}"
FREEZE="${FREEZE:-true}"  # "true" or "false"

SCRATCH="/athena/masonlab/scratch/users/jak4013"
CONDA_ENV="negbiodb-ml"

# === Setup ===
set -euo pipefail
mkdir -p logs

echo "=== Negative-JEPA Fine-tuning ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Pretrained: ${PRETRAINED}"
echo "Domain: ${DOMAIN} | Split: ${SPLIT}"
echo "Config: ${CONFIG} | Freeze encoder: ${FREEZE}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

source /home/fs01/jak4013/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
export PYTHONPATH=src:${PYTHONPATH:-}

# === Build command ===
FREEZE_FLAG="--freeze-encoder"
if [ "${FREEZE}" = "false" ]; then
    FREEZE_FLAG="--unfreeze-encoder"
fi

# --split-col defaults to split_{SPLIT} inside the script; override with SPLIT_COL env var
SPLIT_COL_FLAG=""
if [ -n "${SPLIT_COL:-}" ]; then
    SPLIT_COL_FLAG="--split-col ${SPLIT_COL}"
fi

OUTPUT_DIR_FLAG=""
if [ -n "${OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR_FLAG="--output-dir ${OUTPUT_DIR}"
fi

export PYTHONUNBUFFERED=1

python scripts_jepa/finetune_negjepa.py \
    --pretrained "${PRETRAINED}" \
    --domain "${DOMAIN}" \
    --split "${SPLIT}" \
    --config "${CONFIG}" \
    ${FREEZE_FLAG} \
    ${SPLIT_COL_FLAG} \
    ${OUTPUT_DIR_FLAG}

echo "=== Done: $(date) ==="

# Example usage — submit all cold splits for DTI:
# PRETRAINED=/athena/.../negjepa_pretrain_3d_sigreg/best.pt
# for SPLIT in random cold_compound cold_target cold_both scaffold; do
#   sbatch --export=ALL,PRETRAINED=$PRETRAINED,DOMAIN=dti,SPLIT=$SPLIT \
#     slurm/run_negjepa_finetune.slurm
# done
