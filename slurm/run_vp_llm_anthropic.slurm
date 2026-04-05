#!/bin/bash
#SBATCH --job-name=vp_llm_haiku
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/vp_llm_haiku_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/vp_llm_haiku_%j.err

# Run VP LLM benchmark with Anthropic API (Claude Haiku).
# Usage:
#   sbatch --export=ALL,TASK=vp-l1,CONFIG=zero-shot,FEWSHOT_SET=0 slurm/run_vp_llm_anthropic.slurm
#   sbatch --export=ALL,TASK=vp-l2,CONFIG=3-shot,FEWSHOT_SET=1 slurm/run_vp_llm_anthropic.slurm

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

TASK="${TASK:-vp-l1}"
MODEL="${MODEL:-claude-haiku-4-5-20251001}"
CONFIG="${CONFIG:-zero-shot}"
FEWSHOT_SET="${FEWSHOT_SET:-0}"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONPATH="${PROJECT_DIR}/src"

if [[ -f "${HOME}/.api_keys" ]]; then
    source "${HOME}/.api_keys"
fi

echo "=== VP LLM Anthropic Benchmark ==="
echo "Task: ${TASK}, Model: ${MODEL}, Config: ${CONFIG}, FewshotSet: ${FEWSHOT_SET}"
echo "Start: $(date)"

python "${PROJECT_DIR}/scripts_vp/run_vp_llm_benchmark.py" \
    --task "${TASK}" \
    --config "${CONFIG}" \
    --fewshot-set "${FEWSHOT_SET}" \
    --model "${MODEL}" \
    --provider anthropic \
    --data-dir "${PROJECT_DIR}/exports/vp_llm" \
    --output-dir "${PROJECT_DIR}/results/vp_llm"

echo "End: $(date)"
