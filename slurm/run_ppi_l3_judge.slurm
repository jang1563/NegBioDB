#!/bin/bash
#SBATCH --job-name=ppi_l3_judge
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/ppi_l3_judge_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/ppi_l3_judge_%j.err

# Run PPI L3 LLM-as-Judge evaluation with Gemini API.
# Judges all completed ppi-l3_* runs in results/ppi_llm/.
#
# Usage:
#   sbatch slurm/run_ppi_l3_judge.slurm

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONUNBUFFERED=1

# Load API keys
if [[ -f "${HOME}/.api_keys" ]]; then
    source "${HOME}/.api_keys"
fi

echo "=== PPI L3 Judge Pipeline ==="
echo "Judge model: gemini-2.5-flash"
echo "Node: $(hostname)"
echo "Start: $(date)"

python "${PROJECT_DIR}/scripts_ppi/run_ppi_l3_judge.py" \
    --judge-provider gemini \
    --judge-model gemini-2.5-flash \
    --results-dir "${PROJECT_DIR}/results/ppi_llm"

echo "Done: $(date)"
