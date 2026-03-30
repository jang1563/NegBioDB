#!/bin/bash
#SBATCH --job-name=l3_judge
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=${SCRATCH_DIR:-/path/to/scratch}/negbiodb/logs/l3_judge_%j.log
#SBATCH --error=${SCRATCH_DIR:-/path/to/scratch}/negbiodb/logs/l3_judge_%j.err

# Run L3 LLM-as-Judge evaluation with Gemini API.
# Usage:
#   sbatch slurm/run_l3_judge.slurm                              # Judge all L3 runs
#   sbatch --export=ALL,RUN=l3_llama70b_3-shot_fs0 slurm/run_l3_judge.slurm  # Single run

set -euo pipefail

# Use scratch conda env copy to bypass NFS /home degradation
SCRATCH="${SCRATCH_DIR:-/path/to/scratch}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

# Optional: single run
RUN="${RUN:-}"

# Init — use scratch env via PATH (no conda activate, no NFS /home)
export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"
export PYTHONUNBUFFERED=1

echo "=== L3 Judge Pipeline ==="
echo "Judge model: gemini-2.5-flash-lite"
echo "Node: $(hostname)"
echo "Start: $(date)"

if [ -n "${RUN}" ]; then
    python "${PROJECT_DIR}/scripts/run_l3_judge.py" \
        --run-dir "${PROJECT_DIR}/results/llm/${RUN}" \
        --judge-provider gemini \
        --judge-model gemini-2.5-flash-lite \
        --results-dir "${PROJECT_DIR}/results/llm"
else
    python "${PROJECT_DIR}/scripts/run_l3_judge.py" \
        --judge-provider gemini \
        --judge-model gemini-2.5-flash-lite \
        --results-dir "${PROJECT_DIR}/results/llm"
fi

echo "Done: $(date)"
