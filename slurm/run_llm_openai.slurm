#!/bin/bash
#SBATCH --job-name=llm_openai
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=${SCRATCH_DIR:-/path/to/scratch}/negbiodb/logs/llm_openai_%j.log
#SBATCH --error=${SCRATCH_DIR:-/path/to/scratch}/negbiodb/logs/llm_openai_%j.err

# Run LLM benchmark with OpenAI API (CPU only, fast with high rate limits).
# Usage:
#   sbatch --export=ALL,TASK=l1,MODEL=gpt-4o-mini,CONFIG=zero-shot,FS=0 slurm/run_llm_openai.slurm

set -euo pipefail

# Use scratch conda env copy to bypass NFS /home degradation
SCRATCH="${SCRATCH_DIR:-/path/to/scratch}"
SCRATCH_ENV="${SCRATCH}/conda_env/negbiodb-llm"
PROJECT_DIR="${SCRATCH}/negbiodb"

# Defaults
TASK="${TASK:-l1}"
MODEL="${MODEL:-gpt-4o-mini}"
CONFIG="${CONFIG:-zero-shot}"
FS="${FS:-0}"

# Init — use scratch env via PATH (no conda activate, no NFS /home)
export PATH="${SCRATCH_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SCRATCH_ENV}"

echo "=== LLM OpenAI Benchmark ==="
echo "Task: ${TASK}, Model: ${MODEL}, Config: ${CONFIG}, FS: ${FS}"
echo "Node: $(hostname)"
echo "Start: $(date)"

python "${PROJECT_DIR}/scripts/run_llm_benchmark.py" \
    --task "${TASK}" \
    --model "${MODEL}" \
    --provider openai \
    --config "${CONFIG}" \
    --fewshot-set "${FS}" \
    --data-dir "${PROJECT_DIR}/exports/llm_benchmarks" \
    --output-dir "${PROJECT_DIR}/results/llm"

echo "Done: $(date)"
