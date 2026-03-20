#!/bin/bash
# Submit CT LLM Gemini jobs SEQUENTIALLY to avoid 250 RPD contention.
#
# The free tier only allows 250 requests/day. Running 16 jobs concurrently
# means each gets ~15 RPD (useless). Sequential gives each job the full
# 250 RPD quota. The benchmark script auto-resumes from existing predictions.
#
# Order: smallest tasks first (L3=160 → L4=400 → L2=400 → L1=900)
# Expected timeline at 250 RPD: ~30 days total
#
# Usage:
#   bash slurm/submit_ct_gemini_sequential.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SBATCH="/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch"
SLURM_TMPL="${SCRIPT_DIR}/run_ct_llm_gemini.slurm"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN ==="
fi

MODEL="gemini-2.5-flash"

# Job list: smallest tasks first for fastest partial results
# Format: TASK CONFIG FS
JOBS=(
    "ct-l3 zero-shot 0"
    "ct-l3 3-shot 0"
    "ct-l3 3-shot 1"
    "ct-l3 3-shot 2"
    "ct-l4 zero-shot 0"
    "ct-l4 3-shot 0"
    "ct-l4 3-shot 1"
    "ct-l4 3-shot 2"
    "ct-l2 zero-shot 0"
    "ct-l2 3-shot 0"
    "ct-l2 3-shot 1"
    "ct-l2 3-shot 2"
    "ct-l1 zero-shot 0"
    "ct-l1 3-shot 0"
    "ct-l1 3-shot 1"
    "ct-l1 3-shot 2"
)

PREV_JOBID=""
TOTAL=0

for entry in "${JOBS[@]}"; do
    read -r TASK CONFIG FS <<< "$entry"
    JOB_NAME="ct_${TASK}_${MODEL}_${CONFIG}_fs${FS}"
    TOTAL=$((TOTAL + 1))

    if $DRY_RUN; then
        if [[ -n "$PREV_JOBID" ]]; then
            echo "  [DRY] $JOB_NAME (after job $PREV_JOBID)"
        else
            echo "  [DRY] $JOB_NAME (first job)"
        fi
        PREV_JOBID="DRY_${TOTAL}"
    else
        DEP_FLAG=""
        if [[ -n "$PREV_JOBID" ]]; then
            DEP_FLAG="--dependency=afterany:${PREV_JOBID}"
        fi

        JOBID=$($SBATCH \
            --job-name="$JOB_NAME" \
            --parsable \
            $DEP_FLAG \
            --export="ALL,TASK=$TASK,MODEL=$MODEL,CONFIG=$CONFIG,FS=$FS" \
            "$SLURM_TMPL")

        echo "  Submitted: $JOB_NAME → $JOBID (dep: ${PREV_JOBID:-none})"
        PREV_JOBID="$JOBID"
    fi
done

echo ""
echo "Total jobs: $TOTAL (sequential chain)"
if $DRY_RUN; then
    echo "(Dry run — no jobs submitted)"
    echo "Estimated time at 250 RPD: ~30 days"
else
    echo "Chain started. Each job runs after the previous completes."
    echo "Monitor: squeue -u \$USER | grep gemini"
fi
