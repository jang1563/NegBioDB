#!/bin/bash
#SBATCH --job-name=dc-train-gpu
#SBATCH --partition=scu-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/dc_train_gpu_%j.log
#SBATCH --error=logs/dc_train_gpu_%j.err

# Train DC baselines (DeepSynergy/DrugCombGNN) — GPU required
# Usage: sbatch --export=MODEL=deepsynergy,TASK=m1,SPLIT=random,SEED=42 slurm/run_dc_train_gpu.slurm

set -euo pipefail

MODEL="${MODEL:-deepsynergy}"
TASK="${TASK:-m1}"
SPLIT="${SPLIT:-random}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-50}"

echo "=== DC GPU Training: model=$MODEL task=$TASK split=$SPLIT seed=$SEED ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

cd /athena/masonlab/scratch/users/jak4013/NegBioDB
source slurm/setup_env.sh

mkdir -p logs

PYTHONPATH=src python scripts_dc/train_dc_baseline.py \
    --model "$MODEL" \
    --task "$TASK" \
    --split "$SPLIT" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --device cuda

echo "=== Done: $(date) ==="
