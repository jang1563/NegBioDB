#!/bin/bash
#SBATCH --job-name=ppi_llm_anthropic
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --output=${SCRATCH_DIR:-/path/to/scratch}/negbiodb/logs/ppi_llm_anthropic_%j.log
#SBATCH --error=${SCRATCH_DIR:-/path/to/scratch}/negbiodb/logs/ppi_llm_anthropic_%j.err

# Run PPI LLM benchmark with Anthropic API (CPU only).
# Usage:
#   sbatch --export=ALL,TASK=ppi-l1,MODEL=claude-haiku-4-5,CONFIG=zero-shot,FS=0 slurm/run_ppi_llm_anthropic.slurm

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/path/to/scratch}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

# Defaults
TASK="${TASK:-ppi-l1}"
MODEL="${MODEL:-claude-haiku-4-5}"
CONFIG="${CONFIG:-zero-shot}"
FS="${FS:-0}"
EXTRA="${EXTRA:-}"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"

# Load API keys
if [[ -f "${HOME}/.api_keys" ]]; then
    source "${HOME}/.api_keys"
fi

echo "=== PPI LLM Anthropic Benchmark ==="
echo "Task: ${TASK}, Model: ${MODEL}, Config: ${CONFIG}, FS: ${FS}"
echo "Node: $(hostname)"
echo "Start: $(date)"

python "${PROJECT_DIR}/scripts_ppi/run_ppi_llm_benchmark.py" \
    --task "${TASK}" \
    --model "${MODEL}" \
    --provider anthropic \
    --config "${CONFIG}" \
    --fewshot-set "${FS}" \
    --data-dir "${PROJECT_DIR}/exports/ppi_llm" \
    --output-dir "${PROJECT_DIR}/results/ppi_llm" \
    ${EXTRA}

echo "Done: $(date)"
