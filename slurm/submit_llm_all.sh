#!/bin/bash
# Submit all 72 LLM benchmark runs.
# Phase 1: Local models (GPU) — Llama 70B + Mistral 7B
# Phase 2: Gemini (CPU) — Flash + Flash-Lite
#
# Usage: bash slurm/submit_llm_all.sh

set -euo pipefail

SBATCH="/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch"
SCRATCH="/athena/masonlab/scratch/users/jak4013"
SLURM_DIR="${SCRATCH}/negbiodb/slurm"

TASKS="l1 l4 l3"  # L2 deferred (needs gold annotations)
CONFIGS="zero-shot 3-shot"
FS_SETS="0 1 2"

echo "=== Submitting LLM Benchmark Jobs ==="

# ── Phase 1: Local models (GPU) ──
echo ""
echo "--- Phase 1: Local GPU models ---"
for MODEL in llama70b mistral7b; do
    for TASK in ${TASKS}; do
        for CONFIG in ${CONFIGS}; do
            for FS in ${FS_SETS}; do
                JOB_NAME="llm_${TASK}_${MODEL}_${CONFIG}_fs${FS}"
                echo "Submitting: ${JOB_NAME}"
                ${SBATCH} \
                    --job-name="${JOB_NAME}" \
                    --export=ALL,TASK=${TASK},MODEL=${MODEL},CONFIG=${CONFIG},FS=${FS} \
                    "${SLURM_DIR}/run_llm_local.slurm"
            done
        done
    done
done

# ── Phase 2: Gemini API (CPU, rate-limited) ──
echo ""
echo "--- Phase 2: Gemini API models ---"
for MODEL in gemini-2.5-flash gemini-2.5-flash-lite; do
    for TASK in ${TASKS}; do
        for CONFIG in ${CONFIGS}; do
            for FS in ${FS_SETS}; do
                JOB_NAME="llm_${TASK}_${MODEL}_${CONFIG}_fs${FS}"
                echo "Submitting: ${JOB_NAME}"
                ${SBATCH} \
                    --job-name="${JOB_NAME}" \
                    --export=ALL,TASK=${TASK},MODEL=${MODEL},CONFIG=${CONFIG},FS=${FS} \
                    "${SLURM_DIR}/run_llm_gemini.slurm"
            done
        done
    done
done

echo ""
echo "=== All jobs submitted ==="
echo "Monitor: /opt/ohpc/pub/software/slurm/24.05.2/bin/squeue -u jak4013"
