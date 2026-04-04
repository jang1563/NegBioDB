#!/bin/bash
#SBATCH --job-name=l2_preanno
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/l2_preannotate_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/negbiodb/logs/l2_preannotate_%j.err

# Pre-annotate L2 candidates with GPT-4o-mini for human correction.
# 116 candidates, ~2 min total with OpenAI API.
# Usage:
#   sbatch slurm/run_l2_preannotate.slurm

set -euo pipefail

SCRATCH="${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONUNBUFFERED=1

echo "=== L2 Pre-annotation ==="
echo "Node: $(hostname)"
echo "Start: $(date)"

python "${PROJECT_DIR}/scripts/preannotate_l2.py" \
    --provider openai \
    --model gpt-4o-mini \
    --max-tokens 2048

echo "Done: $(date)"
