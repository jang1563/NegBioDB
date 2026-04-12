#!/bin/bash
#SBATCH --job-name=md_llm_api
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/md_llm_api_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/md_llm_api_%j.err

# MD LLM benchmark via API (Gemini, OpenAI, Anthropic)
# Usage: sbatch --export=ALL,LEVEL=l1,MODEL=gemini-2.5-flash,SHOT=0shot slurm/run_md_llm_api.slurm

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

LEVEL="${LEVEL:-l1}"
MODEL="${MODEL:-gemini-2.5-flash}"
SHOT="${SHOT:-0shot}"
FEWSHOT_SET="${FEWSHOT_SET:-}"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONPATH="${PROJECT_DIR}/src"

if [[ -f "${HOME}/.api_keys" ]]; then
    source "${HOME}/.api_keys"
fi

echo "=== MD LLM API Benchmark ==="
echo "Level: ${LEVEL}, Model: ${MODEL}, Shot: ${SHOT}"
echo "Start: $(date)"

FEWSHOT_ARG=""
if [[ -n "${FEWSHOT_SET}" ]]; then
    FEWSHOT_ARG="--fewshot-set ${FEWSHOT_SET}"
fi

python "${PROJECT_DIR}/scripts_md/08_run_llm_eval.py" \
    --model "${MODEL}" \
    --level "${LEVEL}" \
    --input-dir "${PROJECT_DIR}/exports/md_llm" \
    --output-dir "${PROJECT_DIR}/results/md_llm" \
    ${FEWSHOT_ARG}

echo "End: $(date)"
