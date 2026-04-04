#!/bin/bash
# Submit all 18 NegBioDB PPI baseline training jobs to Cayuga SLURM.
#
# Usage (from Cayuga login node):
#   bash slurm/submit_ppi_all.sh
#   SEEDS="42 43 44" bash slurm/submit_ppi_all.sh
#   MODELS="siamese_cnn pipr" SPLITS="random cold_protein" bash slurm/submit_ppi_all.sh
#
# Job breakdown:
#   B1-B9  : 3 models x 3 splits (random, cold_protein, cold_both) + NegBioDB neg
#   E1-1~6 : 3 models x 2 random conditions (uniform_random, degree_matched)
#   E4-1~3 : 3 models x 1 DDB split + NegBioDB neg
#   Total  : 18 x (#seeds) jobs

set -euo pipefail

SBATCH=${SBATCH_BIN:-/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch}
NEGBIODB=${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}/negbiodb
LOGDIR=$NEGBIODB/logs
SCRIPT=$NEGBIODB/slurm/train_ppi_baseline.slurm
SEEDS_STR=${SEEDS:-42}
MODELS_STR=${MODELS:-"siamese_cnn pipr mlp_features"}
DATASETS_STR=${DATASETS:-balanced}
BASELINE_SPLITS_STR=${SPLITS:-"random cold_protein cold_both ddb"}
NEGATIVES_STR=${NEGATIVES:-"negbiodb uniform_random degree_matched"}

mkdir -p "$LOGDIR"

contains_word() {
    local needle=$1 haystack=$2
    for item in $haystack; do
        if [[ "$item" == "$needle" ]]; then
            return 0
        fi
    done
    return 1
}

# Verify the script exists
if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: SLURM script not found: $SCRIPT"
    exit 1
fi

submit() {
    local model=$1 split=$2 neg=$3 dataset=${4:-balanced} seed=${5:-42}
    local name="ppi_${model}_${dataset}_${split}_${neg}_seed${seed}"
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
    echo "Submitted $name -> job $job_id"
}

echo "=== Submitting NegBioDB PPI baseline jobs ==="
echo "Timestamp: $(date)"
echo "Seeds: $SEEDS_STR"
echo "Models: $MODELS_STR"
echo "Datasets: $DATASETS_STR"
echo "Splits: $BASELINE_SPLITS_STR"
echo "Negatives: $NEGATIVES_STR"
echo ""

# ---- Baselines: 3 models x 3 splits (B1-B9) --------------------------------
echo "--- Baselines (9 runs) ---"
for model in $MODELS_STR; do
    for dataset in $DATASETS_STR; do
        for seed in $SEEDS_STR; do
            if contains_word "negbiodb" "$NEGATIVES_STR"; then
                for split in random cold_protein cold_both; do
                    if contains_word "$split" "$BASELINE_SPLITS_STR"; then
                        submit "$model" "$split" "negbiodb" "$dataset" "$seed"
                    fi
                done
            fi
        done
    done
done

# ---- Exp 1: random negative controls (E1-1 ~ E1-6) -------------------------
echo ""
echo "--- Exp 1: random negatives (6 runs) ---"
if contains_word "balanced" "$DATASETS_STR"; then
    for model in $MODELS_STR; do
        for seed in $SEEDS_STR; do
            for neg in uniform_random degree_matched; do
                if contains_word "random" "$BASELINE_SPLITS_STR" && contains_word "$neg" "$NEGATIVES_STR"; then
                    submit "$model" "random" "$neg" "balanced" "$seed"
                fi
            done
        done
    done
else
    echo "Skipping Exp 1 submissions because DATASETS does not include balanced."
fi

# ---- Exp 4: DDB split (E4-1 ~ E4-3) ----------------------------------------
echo ""
echo "--- Exp 4: DDB split (3 runs) ---"
if contains_word "balanced" "$DATASETS_STR"; then
    for model in $MODELS_STR; do
        for seed in $SEEDS_STR; do
            if contains_word "ddb" "$BASELINE_SPLITS_STR" && contains_word "negbiodb" "$NEGATIVES_STR"; then
                submit "$model" "ddb" "negbiodb" "balanced" "$seed"
            fi
        done
    done
else
    echo "Skipping Exp 4 submissions because DATASETS does not include balanced."
fi

echo ""
echo "=== Submission complete ==="
echo "Monitor with:"
echo "  ${SQUEUE_BIN:-/opt/ohpc/pub/software/slurm/24.05.2/bin/squeue} -u $USER"
echo "  tail -f $LOGDIR/ppi_siamese_cnn_balanced_random_negbiodb_seed${SEEDS_STR%% *}_*.err"
