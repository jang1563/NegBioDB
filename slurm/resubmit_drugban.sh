#!/bin/bash
# Resubmit 3 DrugBAN jobs that timed out at 8h.
# Uses --time=16:00:00 override (scu-gpu max = 48h).
# Optional:
#   SEEDS="42 43" bash slurm/resubmit_drugban.sh

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
            --time=16:00:00 \
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

echo "=== Resubmitting 3 DrugBAN jobs (--time=16:00:00) ==="
echo "Timestamp: $(date)"
echo "Seeds: $SEEDS_STR"
echo ""

for seed in $SEEDS_STR; do
    submit "drugban" "random"        "negbiodb"       "balanced" "$seed"
    submit "drugban" "cold_compound" "negbiodb"       "balanced" "$seed"
    submit "drugban" "random"        "uniform_random" "balanced" "$seed"
done

echo ""
echo "=== Done ==="
