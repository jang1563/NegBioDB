#!/bin/bash
#SBATCH --job-name=ct_l3_judge
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/ct_l3_judge_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/ct_l3_judge_%j.err

# Run CT L3 LLM-as-Judge evaluation with OpenAI API (gpt-4o-mini).
# Judges all completed ct-l3_* runs in results/ct_llm/.

set -euo pipefail

SCRATCH="/athena/masonlab/scratch/users/jak4013"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONUNBUFFERED=1

# Load API keys
if [[ -f "${HOME}/.api_keys" ]]; then
    source "${HOME}/.api_keys"
fi

echo "=== CT L3 Judge Pipeline ==="
echo "Judge model: gpt-4o-mini (OpenAI)"
echo "Node: $(hostname)"
echo "Start: $(date)"

python "${PROJECT_DIR}/scripts_ct/run_ct_l3_judge.py" \
    --judge-provider openai \
    --judge-model gpt-4o-mini \
    --results-dir "${PROJECT_DIR}/results/ct_llm"

echo "Done: $(date)"
