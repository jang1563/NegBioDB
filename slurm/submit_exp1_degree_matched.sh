#!/bin/bash
# Submit Exp 1 degree_matched jobs: 3 models × random split × degree_matched neg × seeds.

set -euo pipefail

SBATCH=${SBATCH_BIN:-/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch}
NEGBIODB=${SCRATCH_DIR:-/path/to/scratch}/negbiodb
LOGDIR=$NEGBIODB/logs
SCRIPT=$NEGBIODB/slurm/train_baseline.slurm
SEEDS_STR=${SEEDS:-42}

submit() {
    local model=$1 split=$2 neg=$3 dataset=${4:-balanced} seed=${5:-42}
    local name="negbio_${model}_${dataset}_${split}_${neg}_seed${seed}"
    local sbatch_out job_id
    sbatch_out=$(
        "$SBATCH" \
            --job-name="$name" \
            --output="$LOGDIR/${name}_%j.out" \
            --error="$LOGDIR/${name}_%j.err" \
            --export=MODEL="$model",SPLIT="$split",NEG="$neg",DATASET="$dataset",SEED="$seed" \
            "$SCRIPT" 2>&1
    )
    job_id=$(echo "$sbatch_out" | grep -oP 'batch job \K\d+')
    if [[ -z "$job_id" ]]; then
        echo "ERROR: Failed to submit $name"
        echo "$sbatch_out"
        exit 1
    fi
    echo "Submitted $name → job $job_id"
}

echo "=== Submitting Exp 1 degree_matched jobs ==="
echo "Timestamp: $(date)"
echo "Seeds: $SEEDS_STR"
echo ""

for model in deepdta graphdta drugban; do
    for seed in $SEEDS_STR; do
        submit "$model" "random" "degree_matched" "balanced" "$seed"
    done
done

echo ""
echo "=== Done ==="
