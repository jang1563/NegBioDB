#!/bin/bash
# NegBioDB-CT baseline training job.
# Job name, output/error paths are set by submit_ct_all.sh.
#
# Required environment variables (set via --export in submit_ct_all.sh):
#   MODEL     : xgboost | mlp | gnn
#   TASK      : m1 | m2
#   SPLIT     : random | cold_drug | cold_condition | temporal | scaffold | degree_balanced
#   NEG       : negbiodb | uniform_random | degree_matched
#   DATASET   : balanced | realistic | smiles_only (M1 only)
#
# Optional:
#   EPOCHS    : number of training epochs (default: 100)
#   PATIENCE  : early stopping patience (default: 15)
#   BATCH     : batch size (default: 256)
#   LR        : learning rate (default: 0.001)
#   SEED      : random seed (default: 42)

# Resource requests are set entirely by submit_ct_all.sh per model type.
# XGBoost: CPU partition, no GPU. MLP/GNN: GPU partition with A40.
# No #SBATCH resource defaults here except --time to avoid conflicts.
#SBATCH --time=6:00:00

# ---------- paths -----------------------------------------------------------
NEGBIODB=/athena/masonlab/scratch/users/jak4013/negbiodb
CONDA_SH=/home/fs01/jak4013/miniconda3/miniconda3/etc/profile.d/conda.sh

# ---------- validate required variables -------------------------------------
if [[ -z "${MODEL:-}" ]] || [[ -z "${TASK:-}" ]] || [[ -z "${SPLIT:-}" ]]; then
    echo "ERROR: Required variables not set. Expected MODEL, TASK, SPLIT via --export."
    echo "  MODEL=${MODEL:-<unset>}  TASK=${TASK:-<unset>}  SPLIT=${SPLIT:-<unset>}"
    exit 1
fi

# ---------- defaults --------------------------------------------------------
NEG=${NEG:-negbiodb}
DATASET=${DATASET:-balanced}
EPOCHS=${EPOCHS:-100}
PATIENCE=${PATIENCE:-15}
BATCH=${BATCH:-256}
LR=${LR:-0.001}
SEED=${SEED:-42}

# ---------- env setup -------------------------------------------------------
source "$CONDA_SH"
conda activate negbiodb-ml

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Model=$MODEL  Task=$TASK  Split=$SPLIT  Neg=$NEG  Dataset=$DATASET  Seed=$SEED"

cd "$NEGBIODB"

# ---------- train -----------------------------------------------------------
python scripts_ct/train_ct_baseline.py \
    --model     "$MODEL"    \
    --task      "$TASK"     \
    --split     "$SPLIT"    \
    --negative  "$NEG"      \
    --dataset   "$DATASET"  \
    --epochs    "$EPOCHS"   \
    --patience  "$PATIENCE" \
    --batch_size "$BATCH"   \
    --lr        "$LR"       \
    --seed      "$SEED"     \
    --data_dir  "$NEGBIODB/exports/ct" \
    --output_dir "$NEGBIODB/results/ct_baselines"

EXIT_CODE=$?
echo "Job finished: $(date) | exit_code=$EXIT_CODE"
exit $EXIT_CODE
