#!/bin/bash
# Submit all GE ML baseline training jobs on Cayuga.
#
# Grid:
#   B1-B10 : 2 models × 5 splits (random, cold_gene, cold_cell_line, cold_both, degree_balanced)
#   E1-1~4 : 2 models × 2 neg controls (uniform_random, degree_matched) × random split
#   Total  : 14 × (#seeds) jobs
#
# Usage:
#   bash slurm/submit_ge_ml_all.sh [--dry-run]
#   SEEDS="42 43 44" bash slurm/submit_ge_ml_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SBATCH="/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch"
SCRIPT="${SCRIPT_DIR}/train_ge_baseline.slurm"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN ==="
fi

SEEDS_STR="${SEEDS:-42}"
MODELS="xgboost mlp"
TASKS="m1"
SPLITS="random cold_gene cold_cell_line cold_both degree_balanced"

TOTAL=0
SUBMITTED=0

submit() {
    local task=$1 model=$2 split=$3 neg=$4 seed=$5 balanced=${6:-}
    local name="ge_${task}_${model}_${split}_${neg}_s${seed}"

    TOTAL=$((TOTAL + 1))

    if $DRY_RUN; then
        echo "  [DRY] $name"
        return
    fi

    echo "  Submitting: $name"
    local bal_flag=""
    if [[ -n "$balanced" ]]; then
        bal_flag=",BALANCED=1"
    fi

    $SBATCH \
        --job-name="$name" \
        --export="ALL,TASK=$task,MODEL=$model,SPLIT=$split,NEG=$neg,SEED=$seed${bal_flag}" \
        "$SCRIPT"
    SUBMITTED=$((SUBMITTED + 1))
}

echo "=== Submitting GE ML baseline jobs ==="
echo "Seeds: $SEEDS_STR"
echo "Models: $MODELS"
echo "Tasks: $TASKS"
echo ""

# ---- Baselines: 2 models × 5 splits × NegBioDB neg (B1-B10) ----
echo "--- Baselines (10 runs per seed) ---"
for task in $TASKS; do
    for model in $MODELS; do
        for seed in $SEEDS_STR; do
            for split in $SPLITS; do
                submit "$task" "$model" "$split" "negbiodb" "$seed" "1"
            done
        done
    done
done

# ---- Exp 1: random negative controls (4 runs per seed) ----
echo ""
echo "--- Exp 1: negative controls (4 runs per seed) ---"
for task in $TASKS; do
    for model in $MODELS; do
        for seed in $SEEDS_STR; do
            for neg in uniform_random degree_matched; do
                submit "$task" "$model" "random" "$neg" "$seed" "1"
            done
        done
    done
done

echo ""
echo "Total jobs: $TOTAL"
if $DRY_RUN; then
    echo "(Dry run — no jobs submitted)"
else
    echo "Submitted: $SUBMITTED"
fi
