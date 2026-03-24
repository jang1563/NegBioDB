#!/bin/bash
#SBATCH --job-name=ge_llm_anthropic
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/ge_llm_anthropic_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/ge_llm_anthropic_%j.err

# Run GE LLM benchmark with Anthropic API.
# Usage:
#   sbatch --export=ALL,TASK=ge-l1,MODEL=claude-haiku-4-5,CONFIG=zero-shot,FEWSHOT_SET=0 slurm/run_ge_llm_anthropic.slurm

set -euo pipefail

SCRATCH="/athena/masonlab/scratch/users/jak4013"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

TASK="${TASK:-ge-l1}"
MODEL="${MODEL:-claude-haiku-4-5}"
CONFIG="${CONFIG:-zero-shot}"
FEWSHOT_SET="${FEWSHOT_SET:-0}"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONPATH="${PROJECT_DIR}/src"

if [[ -f "${HOME}/.api_keys" ]]; then
    source "${HOME}/.api_keys"
fi

echo "=== GE LLM Anthropic Benchmark ==="
echo "Task: ${TASK}, Model: ${MODEL}, Config: ${CONFIG}, FewshotSet: ${FEWSHOT_SET}"
echo "Node: $(hostname)"
echo "Start: $(date)"

python "${PROJECT_DIR}/scripts_depmap/run_ge_llm_benchmark.py" \
    --task "${TASK}" \
    --config "${CONFIG}" \
    --fewshot-set "${FEWSHOT_SET}" \
    --model "${MODEL}" \
    --provider anthropic \
    --data-dir "${PROJECT_DIR}/exports/ge_llm" \
    --output-dir "${PROJECT_DIR}/results/ge_llm"

echo "End: $(date)"
