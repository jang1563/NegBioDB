#!/bin/bash
# Submit all 72 VP CPU training jobs (XGBoost+MLP x m1_balanced+m1_realistic x 6 splits x 3 seeds)

set -euo pipefail

SCRATCH="/athena/masonlab/scratch/users/jak4013"
PROJECT_DIR="${SCRATCH}/negbiodb"
SLURM="${PROJECT_DIR}/slurm/run_vp_train_cpu.slurm"

MODELS=(xgboost mlp)
DATASETS=(m1_balanced m1_realistic)
SPLITS=(random cold_gene cold_disease cold_both degree_balanced temporal)
SEEDS=(42 43 44)

n=0
for MODEL in "${MODELS[@]}"; do
for DATASET in "${DATASETS[@]}"; do
for SPLIT in "${SPLITS[@]}"; do
for SEED in "${SEEDS[@]}"; do
    /opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch \
        --export=ALL,MODEL=${MODEL},DATASET=${DATASET},SPLIT=${SPLIT},SEED=${SEED} \
        "${SLURM}"
    n=$((n+1))
done
done
done
done

echo "Submitted ${n} VP training jobs"
