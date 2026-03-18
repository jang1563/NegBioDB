#!/bin/bash
# Evaluate a saved best.pt checkpoint on the test set (eval-only, no training).
# Required env vars: MODEL, SPLIT, NEG

#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00

NEGBIODB=/athena/masonlab/scratch/users/jak4013/negbiodb
CONDA_SH=/home/fs01/jak4013/miniconda3/miniconda3/etc/profile.d/conda.sh

if [[ -z "${MODEL:-}" ]] || [[ -z "${SPLIT:-}" ]] || [[ -z "${NEG:-}" ]]; then
    echo "ERROR: Required variables not set. Expected MODEL, SPLIT, NEG via --export."
    exit 1
fi

DATASET=${DATASET:-balanced}
SEED=${SEED:-42}
BATCH=${BATCH:-256}

source "$CONDA_SH"
conda activate negbiodb-ml

echo "Eval started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Model=$MODEL  Split=$SPLIT  Neg=$NEG  Dataset=$DATASET  Seed=$SEED"

cd "$NEGBIODB"

python scripts/eval_checkpoint.py \
    --model      "$MODEL"   \
    --split      "$SPLIT"   \
    --negative   "$NEG"     \
    --dataset    "$DATASET" \
    --batch_size "$BATCH"   \
    --seed       "$SEED"    \
    --data_dir   "$NEGBIODB/exports" \
    --output_dir "$NEGBIODB/results/baselines"

EXIT_CODE=$?
echo "Eval finished: $(date) | exit_code=$EXIT_CODE"
exit $EXIT_CODE
