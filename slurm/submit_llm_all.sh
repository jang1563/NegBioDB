#!/bin/bash
# Submit all DTI LLM benchmark runs.
# Phase 1: Local models (GPU) — Llama 70B + Qwen 32B + Mistral 7B
# Phase 2: Gemini (CPU) — Flash + Flash-Lite
# Phase 3: Anthropic (CPU) — Haiku
# Phase 4: OpenAI (CPU) — GPT-4o-mini
#
# Usage: bash slurm/submit_llm_all.sh
#        PHASES="3" bash slurm/submit_llm_all.sh   # Haiku only

set -euo pipefail

SBATCH="/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch"
SCRATCH="${SCRATCH_DIR:-/path/to/scratch}"
SLURM_DIR="${SCRATCH}/negbiodb/slurm"

TASKS="l1 l4 l3"  # L2 deferred (needs gold annotations)
CONFIGS="zero-shot 3-shot"
FS_SETS="0 1 2"
PHASES="${PHASES:-1 2 3 4}"

contains_word() {
    local needle=$1 haystack=$2
    for item in $haystack; do
        [[ "$item" == "$needle" ]] && return 0
    done
    return 1
}

echo "=== Submitting DTI LLM Benchmark Jobs ==="
echo "Tasks: ${TASKS}"
echo "Phases: ${PHASES}"

# ── Phase 1: Local models (GPU) ──
if contains_word "1" "$PHASES"; then
    echo ""
    echo "--- Phase 1: Local GPU models ---"
    for MODEL in llama70b qwen32b mistral7b; do
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
fi

# ── Phase 2: Gemini API (CPU, rate-limited) ──
if contains_word "2" "$PHASES"; then
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
                        --export=ALL,TASK=${TASK},MODEL=${MODEL},CONFIG=${CONFIG},FS=${FS},MAX_TOKENS=1024 \
                        "${SLURM_DIR}/run_llm_gemini.slurm"
                done
            done
        done
    done
fi

# ── Phase 3: Anthropic API (CPU) ──
if contains_word "3" "$PHASES"; then
    echo ""
    echo "--- Phase 3: Anthropic API (Haiku) ---"
    for TASK in ${TASKS}; do
        for CONFIG in ${CONFIGS}; do
            for FS in ${FS_SETS}; do
                JOB_NAME="llm_${TASK}_haiku_${CONFIG}_fs${FS}"
                echo "Submitting: ${JOB_NAME}"
                ${SBATCH} \
                    --job-name="${JOB_NAME}" \
                    --export=ALL,TASK=${TASK},MODEL=claude-haiku-4-5,CONFIG=${CONFIG},FS=${FS} \
                    "${SLURM_DIR}/run_llm_anthropic.slurm"
            done
        done
    done
fi

# ── Phase 4: OpenAI API (CPU) ──
if contains_word "4" "$PHASES"; then
    echo ""
    echo "--- Phase 4: OpenAI API (GPT-4o-mini) ---"
    for TASK in ${TASKS}; do
        for CONFIG in ${CONFIGS}; do
            for FS in ${FS_SETS}; do
                JOB_NAME="llm_${TASK}_gpt4omini_${CONFIG}_fs${FS}"
                echo "Submitting: ${JOB_NAME}"
                ${SBATCH} \
                    --job-name="${JOB_NAME}" \
                    --export=ALL,TASK=${TASK},MODEL=gpt-4o-mini,CONFIG=${CONFIG},FS=${FS} \
                    "${SLURM_DIR}/run_llm_openai.slurm"
            done
        done
    done
fi

echo ""
echo "=== All jobs submitted ==="
echo "Monitor: /opt/ohpc/pub/software/slurm/24.05.2/bin/squeue -u $USER"
