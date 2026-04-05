#!/bin/bash
# Submit all DC ML baseline training jobs on Cayuga.
#
# Grid:
#   CPU: (xgboost, mlp)       × 6 splits × 2 tasks × 3 seeds = 72  → run_dc_train_cpu.slurm
#   GPU: (deepsynergy, gnn)   × 6 splits × 2 tasks × 3 seeds = 72  → run_dc_train_gpu.slurm
#   Total: 144 jobs
#
# Usage:
#   bash slurm/submit_dc_ml_all.sh [--dry-run]
#   SEEDS="42 43 44" bash slurm/submit_dc_ml_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SBATCH="/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN ==="
fi

SEEDS_STR="${SEEDS:-42 43 44}"
TASKS="m1 m2"
SPLITS="random cold_compound cold_cell_line cold_both scaffold leave_one_tissue_out"

TOTAL=0
SUBMITTED=0

submit() {
    local slurm=$1 task=$2 model=$3 split=$4 seed=$5
    local name="dc_${task}_${model}_${split}_s${seed}"

    TOTAL=$((TOTAL + 1))

    if $DRY_RUN; then
        echo "  [DRY] $name → $slurm"
        return
    fi

    echo "  Submitting: $name"
    "$SBATCH" \
        --job-name="$name" \
        --export="ALL,TASK=$task,MODEL=$model,SPLIT=$split,SEED=$seed" \
        "${SCRIPT_DIR}/$slurm"
    SUBMITTED=$((SUBMITTED + 1))
}

echo "=== Submitting DC ML baseline jobs ==="
echo "Seeds: $SEEDS_STR"
echo "Tasks: $TASKS"
echo ""

# ---- CPU models: xgboost, mlp ----
echo "--- CPU models (xgboost, mlp) ---"
for task in $TASKS; do
    for model in xgboost mlp; do
        for seed in $SEEDS_STR; do
            for split in $SPLITS; do
                submit "run_dc_train_cpu.slurm" "$task" "$model" "$split" "$seed"
            done
        done
    done
done

# ---- GPU models: deepsynergy, gnn ----
echo ""
echo "--- GPU models (deepsynergy, gnn) ---"
for task in $TASKS; do
    for model in deepsynergy gnn; do
        for seed in $SEEDS_STR; do
            for split in $SPLITS; do
                submit "run_dc_train_gpu.slurm" "$task" "$model" "$split" "$seed"
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
