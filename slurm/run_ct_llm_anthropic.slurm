#!/bin/bash
#SBATCH --job-name=ct_llm_anthropic
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/ct_llm_anthropic_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/ct_llm_anthropic_%j.err

# Run CT LLM benchmark with Anthropic API (CPU only).
# Usage:
#   sbatch --export=ALL,TASK=ct-l1,MODEL=claude-sonnet-4-6,CONFIG=zero-shot,FS=0 slurm/run_ct_llm_anthropic.slurm

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

# Defaults
TASK="${TASK:-ct-l1}"
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

echo "=== CT LLM Anthropic Benchmark ==="
echo "Task: ${TASK}, Model: ${MODEL}, Config: ${CONFIG}, FS: ${FS}"
echo "Node: $(hostname)"
echo "Start: $(date)"

python "${PROJECT_DIR}/scripts_ct/run_ct_llm_benchmark.py" \
    --task "${TASK}" \
    --model "${MODEL}" \
    --provider anthropic \
    --config "${CONFIG}" \
    --fewshot-set "${FS}" \
    --data-dir "${PROJECT_DIR}/exports/ct_llm" \
    --output-dir "${PROJECT_DIR}/results/ct_llm" \
    ${EXTRA}

echo "Done: $(date)"
