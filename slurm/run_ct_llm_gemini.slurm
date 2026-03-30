#!/bin/bash
#SBATCH --job-name=ct_llm_gemini
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=${SCRATCH_DIR:-/path/to/scratch}/negbiodb/logs/ct_llm_gemini_%j.log
#SBATCH --error=${SCRATCH_DIR:-/path/to/scratch}/negbiodb/logs/ct_llm_gemini_%j.err

# Run CT LLM benchmark with Gemini API (CPU only, long runtime for rate limits).
# Usage:
#   sbatch --export=ALL,TASK=ct-l1,MODEL=gemini-2.5-flash,CONFIG=zero-shot,FS=0 slurm/run_ct_llm_gemini.slurm

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/path/to/scratch}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

# Defaults
TASK="${TASK:-ct-l1}"
MODEL="${MODEL:-gemini-2.5-flash}"
CONFIG="${CONFIG:-zero-shot}"
FS="${FS:-0}"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONUNBUFFERED=1

echo "=== CT LLM Gemini Benchmark ==="
echo "Task: ${TASK}, Model: ${MODEL}, Config: ${CONFIG}, FS: ${FS}"
echo "Node: $(hostname)"
echo "Start: $(date)"

python "${PROJECT_DIR}/scripts_ct/run_ct_llm_benchmark.py" \
    --task "${TASK}" \
    --model "${MODEL}" \
    --provider gemini \
    --config "${CONFIG}" \
    --fewshot-set "${FS}" \
    --data-dir "${PROJECT_DIR}/exports/ct_llm" \
    --output-dir "${PROJECT_DIR}/results/ct_llm"

echo "Done: $(date)"
