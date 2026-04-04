#!/bin/bash
#SBATCH --job-name=l3_judge_val
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/l3_judge_val_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/l3_judge_val_%j.err

# Run L3 second-judge validation with GPT-4o-mini.
# Judges 3 representative runs (120 items total).
# Usage:
#   sbatch slurm/run_l3_judge_validation.slurm

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONUNBUFFERED=1

echo "=== L3 Judge Validation (GPT-4o-mini) ==="
echo "Node: $(hostname)"
echo "Start: $(date)"

python "${PROJECT_DIR}/scripts/run_l3_judge_validation.py" \
    --judge-provider openai \
    --judge-model gpt-4o-mini \
    --results-dir "${PROJECT_DIR}/results/llm"

echo ""
echo "--- Computing Agreement ---"
python "${PROJECT_DIR}/scripts/compute_judge_agreement.py" \
    --results-dir "${PROJECT_DIR}/results/llm"

echo "Done: $(date)"
