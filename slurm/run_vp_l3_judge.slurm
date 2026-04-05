#!/bin/bash
#SBATCH --job-name=vp_l3_judge
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/vp_l3_judge_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/vp_l3_judge_%j.err

# Run VP L3 LLM-as-Judge evaluation with Gemini API.
# Judges all completed vp-l3_* runs in results/vp_llm/.
#
# Usage:
#   sbatch slurm/run_vp_l3_judge.slurm

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONPATH="${PROJECT_DIR}/src"
export PYTHONUNBUFFERED=1

# Load API keys
if [[ -f "${HOME}/.api_keys" ]]; then
    source "${HOME}/.api_keys"
fi

echo "=== VP L3 Judge Pipeline ==="
echo "Judge model: gemini-2.5-flash"
echo "Node: $(hostname)"
echo "Start: $(date)"

python "${PROJECT_DIR}/scripts_vp/run_vp_l3_judge.py" \
    --judge-provider gemini \
    --judge-model gemini-2.5-flash \
    --results-dir "${PROJECT_DIR}/results/vp_llm"

echo "Done: $(date)"
