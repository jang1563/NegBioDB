#!/bin/bash
# Submit Gemini L1 + L4 benchmark runs (D-1 gap fills).
# L1: 12 runs (2 models × 2 configs × 3 fs) minus 1 existing = 11
# L4: 12 runs (2 models × 2 configs × 3 fs) = 12
# Total: 23 runs
#
# Rate limits: Flash 250 RPD, Flash-Lite 1000 RPD
# L1 has 1600 items → Flash needs ~7 days, Flash-Lite ~2 days
# L4 has 400 items → Flash needs ~2 days, Flash-Lite <1 day
#
# Submit sequentially per model to respect rate limits.
# Usage: bash slurm/submit_gemini_l1_l4.sh

set -euo pipefail

SBATCH="/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch"
SCRATCH="/athena/masonlab/scratch/users/jak4013"
SLURM_DIR="${SCRATCH}/negbiodb/slurm"

CONFIGS="zero-shot 3-shot"
FS_SETS="0 1 2"

echo "=== Submitting Gemini L1 + L4 Benchmark Jobs ==="

# Track previous job ID for dependency chaining
PREV_JOB=""

# Helper: submit with optional dependency on previous job
submit_job() {
    local TASK=$1 MODEL=$2 CONFIG=$3 FS=$4 MAX_TOKENS=$5
    local JOB_NAME="llm_${TASK}_${MODEL}_${CONFIG}_fs${FS}"

    # Skip already-completed run
    local RESULT_DIR="${SCRATCH}/negbiodb/results/llm/${TASK}_${MODEL//./-}_${CONFIG}_fs${FS}"
    if [ -f "${RESULT_DIR}/results.json" ]; then
        echo "SKIP (exists): ${JOB_NAME}"
        return
    fi

    local DEP_FLAG=""
    if [ -n "${PREV_JOB}" ]; then
        DEP_FLAG="--dependency=afterany:${PREV_JOB}"
    fi

    echo "Submitting: ${JOB_NAME}"
    PREV_JOB=$(${SBATCH} --parsable ${DEP_FLAG} \
        --job-name="${JOB_NAME}" \
        --export=ALL,TASK=${TASK},MODEL=${MODEL},CONFIG=${CONFIG},FS=${FS},MAX_TOKENS=${MAX_TOKENS} \
        "${SLURM_DIR}/run_llm_gemini.slurm")
    echo "  Job ID: ${PREV_JOB}"
}

# ── Flash-Lite first (higher RPD = faster) ──
echo ""
echo "--- Gemini 2.0 Flash-Lite (1000 RPD) ---"
PREV_JOB=""

for TASK in l4 l1; do
    for CONFIG in ${CONFIGS}; do
        for FS in ${FS_SETS}; do
            submit_job "${TASK}" "gemini-2.5-flash-lite" "${CONFIG}" "${FS}" 1024
        done
    done
done

# ── Flash (lower RPD, chained sequentially) ──
echo ""
echo "--- Gemini 2.5 Flash (250 RPD) ---"
PREV_JOB=""

for TASK in l4 l1; do
    MAX_TOKENS=1024
    for CONFIG in ${CONFIGS}; do
        for FS in ${FS_SETS}; do
            submit_job "${TASK}" "gemini-2.5-flash" "${CONFIG}" "${FS}" "${MAX_TOKENS}"
        done
    done
done

echo ""
echo "=== All Gemini L1/L4 jobs submitted ==="
echo "Monitor: /opt/ohpc/pub/software/slurm/24.05.2/bin/squeue -u jak4013"
