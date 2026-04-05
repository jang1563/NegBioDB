#!/bin/bash
#SBATCH --job-name=dc_llm_anthropic
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/dc_llm_anthropic_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/dc_llm_anthropic_%j.err

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

TASK="${TASK:-dc-l1}"
MODEL="${MODEL:-claude-haiku-4-5}"
CONFIG="${CONFIG:-zero-shot}"
FEWSHOT_SET="${FEWSHOT_SET:-0}"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONPATH="${PROJECT_DIR}/src"

if [[ -f "${HOME}/.api_keys" ]]; then
    source "${HOME}/.api_keys"
fi

echo "=== DC LLM Anthropic Benchmark ==="
echo "Task: ${TASK}, Model: ${MODEL}, Config: ${CONFIG}, FewshotSet: ${FEWSHOT_SET}"
echo "Start: $(date)"

python "${PROJECT_DIR}/scripts_dc/run_dc_llm_benchmark.py" \
    --task "${TASK}" \
    --config "${CONFIG}" \
    --fewshot-set "${FEWSHOT_SET}" \
    --model "${MODEL}" \
    --provider anthropic \
    --data-dir "${PROJECT_DIR}/exports/dc_llm" \
    --output-dir "${PROJECT_DIR}/results/dc_llm"

echo "End: $(date)"
