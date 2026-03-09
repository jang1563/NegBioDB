#!/bin/bash
# Submit all 18 NegBioDB baseline training jobs to Cayuga SLURM.
#
# Usage (from Cayuga login node, after setup_env.sh + precompute_graphs.sh):
#   bash slurm/submit_all.sh
#
# Job breakdown:
#   B1-B9  : 3 models × 3 splits (random, cold_compound, cold_target) + NegBioDB neg
#   E1-1~6 : 3 models × 2 random conditions (uniform_random, degree_matched)
#   E4-1~3 : 3 models × 1 DDB split + NegBioDB neg
#   Total  : 18 jobs

set -euo pipefail

SBATCH=/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch
NEGBIODB=/athena/masonlab/scratch/users/jak4013/negbiodb
LOGDIR=$NEGBIODB/logs
SCRIPT=$NEGBIODB/slurm/train_baseline.slurm

mkdir -p "$LOGDIR"

# Verify the script exists
if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: SLURM script not found: $SCRIPT"
    exit 1
fi

# Verify the graph cache exists (required for GraphDTA/DrugBAN)
GRAPH_CACHE=$NEGBIODB/exports/graph_cache.pt
if [[ ! -f "$GRAPH_CACHE" ]]; then
    echo "ERROR: Graph cache not found at $GRAPH_CACHE"
    echo "Run precompute_graphs.sh first and wait for it to complete before running submit_all.sh."
    exit 1
fi

submit() {
    local model=$1 split=$2 neg=$3
    local name="negbio_${model}_${split}_${neg}"
    local sbatch_out job_id
    sbatch_out=$(
        "$SBATCH" \
            --job-name="$name" \
            --output="$LOGDIR/${name}_%j.out" \
            --error="$LOGDIR/${name}_%j.err" \
            --export=MODEL="$model",SPLIT="$split",NEG="$neg" \
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

echo "=== Submitting NegBioDB ML baseline jobs ==="
echo "Timestamp: $(date)"
echo ""

# ---- Baselines: 3 models × 3 splits (B1-B9) --------------------------------
echo "--- Baselines (9 runs) ---"
for model in deepdta graphdta drugban; do
    for split in random cold_compound cold_target; do
        submit "$model" "$split" "negbiodb"
    done
done

# ---- Exp 1: random negative controls (E1-1 ~ E1-6) ------------------------
echo ""
echo "--- Exp 1: random negatives (6 runs) ---"
for model in deepdta graphdta drugban; do
    for neg in uniform_random degree_matched; do
        submit "$model" "random" "$neg"
    done
done

# ---- Exp 4: DDB split (E4-1 ~ E4-3) ----------------------------------------
echo ""
echo "--- Exp 4: DDB split (3 runs) ---"
for model in deepdta graphdta drugban; do
    submit "$model" "ddb" "negbiodb"
done

echo ""
echo "=== All 18 jobs submitted ==="
echo "Monitor with:"
echo "  /opt/ohpc/pub/software/slurm/24.05.2/bin/squeue -u jak4013"
echo "  tail -f $LOGDIR/negbio_deepdta_random_negbiodb_*.err"
