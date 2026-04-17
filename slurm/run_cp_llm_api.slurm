#!/bin/bash
#SBATCH --job-name=cp_llm_api
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/cp_llm_api_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/cp_llm_api_%j.err

# CP LLM benchmark via API (Gemini, OpenAI, Anthropic)
# Usage: sbatch --export=ALL,TASK=cp-l1,PROVIDER=gemini,MODEL=gemini-2.5-flash,CONFIG=zero-shot slurm/run_cp_llm_api.slurm

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

TASK="${TASK:-cp-l1}"
PROVIDER="${PROVIDER:-gemini}"
MODEL="${MODEL:-gemini-2.5-flash}"
CONFIG="${CONFIG:-zero-shot}"
FEWSHOT_SET="${FEWSHOT_SET:-0}"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONPATH="${PROJECT_DIR}/src"

if [[ -f "${HOME}/.api_keys" ]]; then
    source "${HOME}/.api_keys"
fi

echo "=== CP LLM API Benchmark ==="
echo "Task: ${TASK}, Provider: ${PROVIDER}, Model: ${MODEL}, Config: ${CONFIG}, FewshotSet: ${FEWSHOT_SET}"
echo "Start: $(date)"

JUDGE_ARGS=""
if [[ "${TASK}" == "cp-l3" ]]; then
    JUDGE_PROVIDER="${JUDGE_PROVIDER:-gemini}"
    JUDGE_MODEL="${JUDGE_MODEL:-gemini-2.5-flash}"
    JUDGE_ARGS="--judge-provider ${JUDGE_PROVIDER} --judge-model ${JUDGE_MODEL}"
    echo "Judge: ${JUDGE_PROVIDER}/${JUDGE_MODEL}"
fi

python "${PROJECT_DIR}/scripts_cp/run_cp_llm_benchmark.py" \
    --task "${TASK}" \
    --config "${CONFIG}" \
    --fewshot-set "${FEWSHOT_SET}" \
    --model "${MODEL}" \
    --provider "${PROVIDER}" \
    --data-dir "${PROJECT_DIR}/exports/cp_llm" \
    --output-dir "${PROJECT_DIR}/results/cp_llm" \
    ${JUDGE_ARGS}

echo "End: $(date)"
