#!/bin/bash
#SBATCH --job-name=negjepa_pretrain
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=47:59:00
#SBATCH --output=logs/negjepa_pretrain_%j.log
#SBATCH --error=logs/negjepa_pretrain_%j.err

# === Configuration (override via sbatch --export) ===
CONFIG="${CONFIG:-configs/negbiojepa/pretrain_3domain.yaml}"
RESUME="${RESUME:-}"  # path to checkpoint to resume from

SCRATCH="/athena/masonlab/scratch/users/jak4013"
CONDA_ENV="negbiodb-ml"

# === Setup ===
set -euo pipefail
mkdir -p logs

echo "=== Negative-JEPA Pretraining ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Config: ${CONFIG}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

source /home/fs01/jak4013/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
export PYTHONPATH=src:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

# === Build command ===
CMD="python scripts_jepa/pretrain_negjepa.py --config ${CONFIG}"

if [ -n "${RESUME}" ]; then
    CMD="${CMD} --resume ${RESUME}"
fi

# === Run ===
eval ${CMD}

echo "=== Done: $(date) ==="
