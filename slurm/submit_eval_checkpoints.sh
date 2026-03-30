#!/bin/bash
# Evaluate timed-out checkpoints:
#   drugban_balanced_random_uniform_random_seed{seed}
#   drugban_balanced_random_degree_matched_seed{seed}
# Optional:
#   SEEDS="42 43" bash slurm/submit_eval_checkpoints.sh

set -euo pipefail

SBATCH=/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch
NEGBIODB=${SCRATCH_DIR:-/path/to/scratch}/negbiodb
LOGDIR=$NEGBIODB/logs
SCRIPT=$NEGBIODB/slurm/eval_checkpoint.slurm
DATASET=${DATASET:-balanced}
SEEDS_STR=${SEEDS:-42}

submit() {
    local model=$1 split=$2 neg=$3 dataset=${4:-balanced} seed=${5:-42}
    local name="negbio_eval_${model}_${dataset}_${split}_${neg}_seed${seed}"
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

echo "=== Submitting eval-only jobs for timed-out checkpoints ==="
echo "Timestamp: $(date)"
echo "Dataset: $DATASET"
echo "Seeds: $SEEDS_STR"
echo ""

for seed in $SEEDS_STR; do
    submit "drugban" "random" "uniform_random" "$DATASET" "$seed"
    submit "drugban" "random" "degree_matched" "$DATASET" "$seed"
done

echo ""
echo "=== Done ==="
