#!/bin/bash
#SBATCH --job-name=dc-train-cpu
#SBATCH --partition=scu-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/dc_train_cpu_%j.log
#SBATCH --error=logs/dc_train_cpu_%j.err

# Train DC baselines (XGBoost/MLP) — CPU only
# Usage: sbatch --export=MODEL=xgboost,TASK=m1,SPLIT=random,SEED=42 slurm/run_dc_train_cpu.slurm

set -euo pipefail

MODEL="${MODEL:-xgboost}"
TASK="${TASK:-m1}"
SPLIT="${SPLIT:-random}"
SEED="${SEED:-42}"

echo "=== DC CPU Training: model=$MODEL task=$TASK split=$SPLIT seed=$SEED ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

cd /athena/masonlab/scratch/users/jak4013/NegBioDB
source slurm/setup_env.sh

mkdir -p logs

PYTHONPATH=src python scripts_dc/train_dc_baseline.py \
    --model "$MODEL" \
    --task "$TASK" \
    --split "$SPLIT" \
    --seed "$SEED" \
    --device cpu

echo "=== Done: $(date) ==="
