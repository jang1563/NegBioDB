#!/bin/bash
# NegBioDB baseline training job.
# Job name, output/error paths are set by submit_all.sh (not here)
# because #SBATCH directives cannot expand --export variables.
#
# Required environment variables (set via --export in submit_all.sh):
#   MODEL     : deepdta | graphdta | drugban
#   SPLIT     : random | cold_compound | cold_target | ddb
#   NEG       : negbiodb | uniform_random | degree_matched
#   DATASET   : balanced | realistic (random controls support balanced only)
#
# Optional:
#   EPOCHS    : number of training epochs (default: 100)
#   PATIENCE  : early stopping patience (default: 10)
#   BATCH     : batch size (default: 256; DrugBAN auto-reduces to 128)
#   LR        : learning rate (default: 0.001)
#   SEED      : random seed (default: 42)

#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
# Uncomment if masonlab reservation is available:
# #SBATCH --reservation=masonlab

# ---------- paths -----------------------------------------------------------
NEGBIODB=/athena/masonlab/scratch/users/jak4013/negbiodb
CONDA_SH=/home/fs01/jak4013/miniconda3/miniconda3/etc/profile.d/conda.sh

# ---------- validate required variables -------------------------------------
if [[ -z "${MODEL:-}" ]] || [[ -z "${SPLIT:-}" ]] || [[ -z "${NEG:-}" ]]; then
    echo "ERROR: Required variables not set. Expected MODEL, SPLIT, NEG via --export."
    echo "  MODEL=${MODEL:-<unset>}  SPLIT=${SPLIT:-<unset>}  NEG=${NEG:-<unset>}"
    exit 1
fi

# ---------- defaults --------------------------------------------------------
DATASET=${DATASET:-balanced}
EPOCHS=${EPOCHS:-100}
PATIENCE=${PATIENCE:-10}
BATCH=${BATCH:-256}
LR=${LR:-0.001}
SEED=${SEED:-42}

# ---------- env setup -------------------------------------------------------
source "$CONDA_SH"
conda activate negbiodb-ml

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Model=$MODEL  Split=$SPLIT  Neg=$NEG  Dataset=$DATASET  Seed=$SEED"

cd "$NEGBIODB"

# ---------- train -----------------------------------------------------------
python scripts/train_baseline.py \
    --model     "$MODEL"    \
    --split     "$SPLIT"    \
    --negative  "$NEG"      \
    --dataset   "$DATASET"  \
    --epochs    "$EPOCHS"   \
    --patience  "$PATIENCE" \
    --batch_size "$BATCH"   \
    --lr        "$LR"       \
    --seed      "$SEED"     \
    --data_dir  "$NEGBIODB/exports" \
    --output_dir "$NEGBIODB/results/baselines"

EXIT_CODE=$?
echo "Job finished: $(date) | exit_code=$EXIT_CODE"
exit $EXIT_CODE
