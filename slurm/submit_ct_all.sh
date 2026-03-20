#!/bin/bash
# Submit all CT baseline training jobs to Cayuga SLURM.
#
# Usage (from Cayuga login node):
#   bash slurm/submit_ct_all.sh
#   SEEDS="42 43 44" bash slurm/submit_ct_all.sh
#   MODELS="xgboost" TASKS="m1" bash slurm/submit_ct_all.sh
#
# Job breakdown (36 per seed × 3 seeds = 108 total):
#   M1 baselines : 3 models × 3 splits (random, cold_drug, cold_condition) = 9
#   M1 Exp CT-1  : 3 models × 1 split (random) × 2 neg sources = 6
#   M1 Exp CT-3  : 3 models × 1 split (temporal) = 3
#   M2           : 3 models × 6 splits = 18

set -euo pipefail

SBATCH=${SBATCH_BIN:-/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch}
NEGBIODB=/athena/masonlab/scratch/users/jak4013/negbiodb
LOGDIR=$NEGBIODB/logs/ct
SCRIPT=$NEGBIODB/slurm/train_ct_baseline.slurm
SEEDS_STR=${SEEDS:-"42 43 44"}
MODELS_STR=${MODELS:-"xgboost mlp gnn"}
TASKS_STR=${TASKS:-"m1 m2"}

mkdir -p "$LOGDIR"

if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: SLURM script not found: $SCRIPT"
    exit 1
fi

submit() {
    local model=$1 task=$2 split=$3 neg=$4 dataset=${5:-balanced} seed=${6:-42}
    local name="ct_${model}_${task}_${split}_${neg}_seed${seed}"

    # Set partition and resources based on model
    local partition gres cpus mem
    if [[ "$model" == "xgboost" ]]; then
        partition="scu-cpu"
        gres=""
        cpus=16
        mem="64G"
    else
        partition="scu-gpu"
        gres="gpu:a40:1"
        cpus=8
        mem="32G"
    fi

    local sbatch_args=(
        --job-name="$name"
        --output="$LOGDIR/${name}_%j.out"
        --error="$LOGDIR/${name}_%j.err"
        --partition="$partition"
        --cpus-per-task="$cpus"
        --mem="$mem"
        --export=MODEL="$model",TASK="$task",SPLIT="$split",NEG="$neg",DATASET="$dataset",SEED="$seed"
    )
    if [[ -n "$gres" ]]; then
        sbatch_args+=(--gres="$gres")
    fi

    local sbatch_out job_id
    sbatch_out=$("$SBATCH" "${sbatch_args[@]}" "$SCRIPT" 2>&1)
    job_id=$(echo "$sbatch_out" | grep -oP 'batch job \K\d+')
    if [[ -z "$job_id" ]]; then
        echo "ERROR: Failed to submit $name"
        echo "$sbatch_out"
        exit 1
    fi
    echo "Submitted $name → job $job_id"
}

echo "=== Submitting NegBioDB-CT ML baseline jobs ==="
echo "Timestamp: $(date)"
echo "Seeds: $SEEDS_STR"
echo "Models: $MODELS_STR"
echo "Tasks: $TASKS_STR"
echo ""

JOB_COUNT=0

# ---- M1 Baselines: 3 models × 3 splits × negbiodb (9 runs) ----------------
if echo "$TASKS_STR" | grep -qw "m1"; then
    echo "--- M1 Baselines (9 runs) ---"
    for model in $MODELS_STR; do
        for seed in $SEEDS_STR; do
            for split in random cold_drug cold_condition; do
                submit "$model" "m1" "$split" "negbiodb" "balanced" "$seed"
                JOB_COUNT=$((JOB_COUNT + 1))
            done
        done
    done

    # ---- M1 Exp CT-1: 3 models × random × 2 neg sources (6 runs) -----------
    echo ""
    echo "--- M1 Exp CT-1: alternative negatives (6 runs) ---"
    for model in $MODELS_STR; do
        for seed in $SEEDS_STR; do
            for neg in uniform_random degree_matched; do
                submit "$model" "m1" "random" "$neg" "balanced" "$seed"
                JOB_COUNT=$((JOB_COUNT + 1))
            done
        done
    done

    # ---- M1 Exp CT-3: 3 models × temporal split (3 runs) --------------------
    echo ""
    echo "--- M1 Exp CT-3: temporal split (3 runs) ---"
    for model in $MODELS_STR; do
        for seed in $SEEDS_STR; do
            submit "$model" "m1" "temporal" "negbiodb" "balanced" "$seed"
            JOB_COUNT=$((JOB_COUNT + 1))
        done
    done
fi

# ---- M2: 3 models × 6 splits (18 runs) ------------------------------------
if echo "$TASKS_STR" | grep -qw "m2"; then
    echo ""
    echo "--- M2: all splits (18 runs) ---"
    for model in $MODELS_STR; do
        for seed in $SEEDS_STR; do
            for split in random cold_drug cold_condition temporal scaffold degree_balanced; do
                submit "$model" "m2" "$split" "negbiodb" "balanced" "$seed"
                JOB_COUNT=$((JOB_COUNT + 1))
            done
        done
    done
fi

echo ""
echo "=== Submission complete: $JOB_COUNT jobs ==="
echo "Monitor with:"
echo "  ${SQUEUE_BIN:-/opt/ohpc/pub/software/slurm/24.05.2/bin/squeue} -u jak4013"
echo "  tail -f $LOGDIR/ct_xgboost_m1_random_negbiodb_seed${SEEDS_STR%% *}_*.err"
