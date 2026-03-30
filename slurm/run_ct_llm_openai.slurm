#!/bin/bash
#SBATCH --job-name=ct_llm_openai
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=${SCRATCH_DIR:-/path/to/scratch}/negbiodb/logs/ct_llm_openai_%j.log
#SBATCH --error=${SCRATCH_DIR:-/path/to/scratch}/negbiodb/logs/ct_llm_openai_%j.err

# Run CT LLM benchmark with OpenAI API (CPU only).
# Usage:
#   sbatch --export=ALL,TASK=ct-l1,MODEL=gpt-4o-mini,CONFIG=zero-shot,FS=0 slurm/run_ct_llm_openai.slurm

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/path/to/scratch}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

# Defaults
TASK="${TASK:-ct-l1}"
MODEL="${MODEL:-gpt-4o-mini}"
CONFIG="${CONFIG:-zero-shot}"
FS="${FS:-0}"
EXTRA="${EXTRA:-}"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"

echo "=== CT LLM OpenAI Benchmark ==="
echo "Task: ${TASK}, Model: ${MODEL}, Config: ${CONFIG}, FS: ${FS}"
echo "Node: $(hostname)"
echo "Start: $(date)"

python "${PROJECT_DIR}/scripts_ct/run_ct_llm_benchmark.py" \
    --task "${TASK}" \
    --model "${MODEL}" \
    --provider openai \
    --config "${CONFIG}" \
    --fewshot-set "${FS}" \
    --data-dir "${PROJECT_DIR}/exports/ct_llm" \
    --output-dir "${PROJECT_DIR}/results/ct_llm" \
    ${EXTRA}

echo "Done: $(date)"
